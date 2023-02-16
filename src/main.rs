use bevy::asset::LoadState;
use bevy::prelude::*;
use bevy::render::mesh::Indices;
use bevy::render::render_resource::{AddressMode, PrimitiveTopology, SamplerDescriptor};
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use block_mesh::ndshape::{ConstShape, ConstShape3u32};
use block_mesh::{
    greedy_quads, visible_block_faces, GreedyQuadsBuffer, MergeVoxel, UnitQuadBuffer,
    Voxel as MeshableVoxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG,
};
use rand::Rng;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum AppState {
    Loading,
    Run,
}

const UV_SCALE: f32 = 1.0 / 16.0;

#[derive(Resource)]
struct Loading(Handle<Image>);

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(WorldInspectorPlugin)
        .insert_resource(State::new(AppState::Loading))
        .add_state(AppState::Loading)
        .add_system_set(SystemSet::on_enter(AppState::Loading).with_system(load_assets))
        .add_system_set(SystemSet::on_update(AppState::Loading).with_system(check_loaded))
        .add_system_set(SystemSet::on_enter(AppState::Run).with_system(setup))
        .add_system_set(SystemSet::on_update(AppState::Run).with_system(camera_rotation_system))
        .run();
}

fn load_assets(mut commands: Commands, asset_server: Res<AssetServer>) {
    debug!("load");
    let handle = asset_server.load("uv_checker.png");
    commands.insert_resource(Loading(handle));
}

/// Make sure that our texture is loaded so we can change some settings on it later
fn check_loaded(
    mut state: ResMut<State<AppState>>,
    handle: Res<Loading>,
    asset_server: Res<AssetServer>,
) {
    debug!("check loaded");
    if let LoadState::Loaded = asset_server.get_load_state(&handle.0) {
        state.set(AppState::Run).unwrap();
    }
}

#[derive(Copy, Clone, Hash, Debug, PartialEq, Eq)]
pub struct Voxel(pub u8);

impl Voxel {
    pub const EMPTY_VOXEL: Voxel = Voxel(0);
    pub const A1_VOXEL: Voxel = Voxel(1);
    pub const A2_VOXEL: Voxel = Voxel(2);
}

impl MergeVoxel for Voxel {
    type MergeValue = u8;
    type MergeValueFacingNeighbour = u8;

    #[inline]
    fn merge_value(&self) -> Self::MergeValue {
        self.0
    }
    #[inline]
    fn merge_value_facing_neighbour(&self) -> Self::MergeValueFacingNeighbour {
        self.0 * 2
    }
}

impl Default for Voxel {
    fn default() -> Self {
        Self::EMPTY_VOXEL
    }
}

impl MeshableVoxel for Voxel {
    #[inline]
    fn get_visibility(&self) -> block_mesh::VoxelVisibility {
        match *self {
            Self::EMPTY_VOXEL => block_mesh::VoxelVisibility::Empty,
            Self::A1_VOXEL => block_mesh::VoxelVisibility::Translucent,
            _ => block_mesh::VoxelVisibility::Opaque,
        }
    }
}

fn ao_convert(ao: Vec<u8>, num_vertices: usize) -> Vec<[f32; 4]> {
    let mut res = Vec::with_capacity(num_vertices);
    for value in ao {
        match value {
            0 => res.extend_from_slice(&[[0.1, 0.1, 0.1, 1.0]]),
            1 => res.extend_from_slice(&[[0.3, 0.3, 0.3, 1.0]]),
            2 => res.extend_from_slice(&[[0.5, 0.5, 0.5, 1.0]]),
            3 => res.extend_from_slice(&[[0.75, 0.75, 0.75, 1.0]]),
            _ => res.extend_from_slice(&[[1., 1., 1., 1.0]]),
        }
    }
    return res;
}

fn setup(
    mut commands: Commands,
    texture_handle: Res<Loading>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    // mut textures: ResMut<Assets<Image>>,
) {
    debug!("setup");
    // let mut texture = textures.get_mut(&texture_handle.0).unwrap();

    type SampleShape = ConstShape3u32<22, 22, 22>;

    // Just a solid cube of voxels. We only fill the interior since we need some empty voxels to form a boundary for the mesh.
    let mut voxels = [Voxel(0); SampleShape::SIZE as usize];
    for z in 1..21 {
        for y in 1..21 {
            for x in 1..21 {
                let i = SampleShape::linearize([x, y, z]);
                let vox_type = rand::thread_rng().gen_range(0..3);
                voxels[i as usize] = Voxel(vox_type);
            }
        }
    }

    let faces = RIGHT_HANDED_Y_UP_CONFIG.faces;

    // Simple meshing works on web and makes texture atlases easier. However I may look into greedy meshing in future
    let mut buffer = UnitQuadBuffer::new();
    visible_block_faces(
        &voxels,
        &SampleShape {},
        [0; 3],
        [21; 3],
        &faces,
        &mut buffer,
    );
    let num_indices = buffer.num_quads() * 6;
    let num_vertices = buffer.num_quads() * 4;
    let mut indices = Vec::with_capacity(num_indices);
    let mut positions = Vec::with_capacity(num_vertices);
    let mut normals = Vec::with_capacity(num_vertices);
    let mut tex_coords = Vec::with_capacity(num_vertices);
    let mut ao = Vec::with_capacity(num_vertices);
    for (group, face) in buffer.groups.into_iter().zip(faces.into_iter()) {
        for quad in group.into_iter() {
            indices.extend_from_slice(&face.quad_mesh_indices(positions.len() as u32));
            positions.extend_from_slice(&face.quad_mesh_positions(&quad.into(), 1.0));
            normals.extend_from_slice(&face.quad_mesh_normals());
            ao.extend_from_slice(&face.quad_mesh_ao(&quad.into()));
            let mut face_tex =
                face.tex_coords(RIGHT_HANDED_Y_UP_CONFIG.u_flip_face, true, &quad.into());
            let [x, y, z] = quad.minimum;
            let i = SampleShape::linearize([x, y, z]);
            let voxel_type = voxels[i as usize];
            let tile_size = 64.0;
            let texture_size = 1024.0;
            match voxel_type {
                Voxel(1) => {
                    let tile_offset = 10.0;
                    face_tex[0][0] = ((tile_offset - 1.0) * tile_size) / texture_size;
                    face_tex[0][1] = ((tile_offset - 1.0) * tile_size) / texture_size;
                    face_tex[1][0] = (tile_offset * tile_size) / texture_size;
                    face_tex[1][1] = ((tile_offset - 1.0) * tile_size) / texture_size;
                    face_tex[2][0] = ((tile_offset - 1.0) * tile_size) / texture_size;
                    face_tex[2][1] = (tile_offset * tile_size) / texture_size;
                    face_tex[3][0] = (tile_offset * tile_size) / texture_size;
                    face_tex[3][1] = (tile_offset * tile_size) / texture_size;
                }
                Voxel(2) => {
                    let tile_offset = 16.0;
                    face_tex[0][0] = ((tile_offset - 1.0) * tile_size) / texture_size;
                    face_tex[0][1] = ((tile_offset - 1.0) * tile_size) / texture_size;
                    face_tex[1][0] = (tile_offset * tile_size) / texture_size;
                    face_tex[1][1] = ((tile_offset - 1.0) * tile_size) / texture_size;
                    face_tex[2][0] = ((tile_offset - 1.0) * tile_size) / texture_size;
                    face_tex[2][1] = (tile_offset * tile_size) / texture_size;
                    face_tex[3][0] = (tile_offset * tile_size) / texture_size;
                    face_tex[3][1] = (tile_offset * tile_size) / texture_size;
                }
                _ => {
                    println!("What");
                }
            }
            tex_coords.extend_from_slice(&face_tex);
        }
    }

    let finalao = ao_convert(ao, num_vertices);
    let mut render_mesh = Mesh::new(PrimitiveTopology::TriangleList);

    render_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    render_mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    render_mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, tex_coords);
    render_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, finalao);
    render_mesh.set_indices(Some(Indices::U32(indices)));

    commands.spawn(PbrBundle {
        mesh: meshes.add(render_mesh.clone()),
        material: materials.add(StandardMaterial {
            base_color: Color::WHITE,
            base_color_texture: Some(texture_handle.0.clone()),
            alpha_mode: AlphaMode::Mask((1.0)),
            perceptual_roughness: 1.0,
            ..default()
        }),
        transform: Transform::from_translation(Vec3::splat(-10.0)),
        ..Default::default()
    });
    commands.spawn(PbrBundle {
        mesh: meshes.add(render_mesh),
        material: materials.add(StandardMaterial {
            base_color: Color::WHITE,
            base_color_texture: Some(texture_handle.0.clone()),
            alpha_mode: AlphaMode::Blend,
            perceptual_roughness: 1.0,
            ..default()
        }),
        transform: Transform::from_translation(Vec3::splat(-10.0)),
        ..Default::default()
    });

    commands.spawn(PointLightBundle {
        transform: Transform::from_translation(Vec3::new(0.0, 50.0, 50.0)),
        point_light: PointLight {
            range: 200.0,
            intensity: 50000.0,
            shadows_enabled: true,
            ..Default::default()
        },
        ..Default::default()
    });
    let camera = commands.spawn(Camera3dBundle::default()).id();
    commands.insert_resource(CameraRotationState::new(camera));
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 0.5,
    });
}

#[derive(Resource)]
struct CameraRotationState {
    camera: Entity,
}

impl CameraRotationState {
    fn new(camera: Entity) -> Self {
        Self { camera }
    }
}

fn camera_rotation_system(
    state: Res<CameraRotationState>,
    time: Res<Time>,
    mut transforms: Query<&mut Transform>,
) {
    let t = 0.3 * time.elapsed_seconds() as f32;

    let target = Vec3::new(0.0, 0.0, 0.0);
    let height = 30.0 * (2.0 * t).sin();
    let radius = 50.0;
    let x = radius * t.cos();
    let z = radius * t.sin();
    let mut eye = Transform::from_translation(Vec3::new(x, height, z));
    eye.look_at(target, Vec3::Y);

    let mut cam_tfm = transforms.get_mut(state.camera).unwrap();
    *cam_tfm = eye;
}
