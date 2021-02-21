bl_info = {
    "name": "Vertex color to texture",
    "author": "AZ",
    "version": (1, 0),
    "blender": (2, 79, 0),
    "warning": "",
    "wiki_url": "",
}

import bpy
from bpy.types import Operator
from bpy.props import FloatVectorProperty
from bpy_extras.object_utils import AddObjectHelper, object_data_add
from mathutils import Vector
import sys

# params: path_to_ply_model, path_to_output_PNG

def register():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]

    # Load model
    bpy.ops.import_mesh.ply(filepath=argv[0])

    # Create image, unwrap the model
    bpy.ops.object.select_all(action='DESELECT')

    bpy.ops.image.new(name="Texture", width=4096, height=4096, color=(0.0, 0.0, 0.0, 0.0), alpha=True,
                      generated_type='BLANK', float=False, gen_context='NONE', use_stereo_3d=False)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.0, user_area_weight=0.0, use_aspect=True,
                             stretch_to_bounds=True)
    bpy.ops.uv.select_all(action='SELECT')

    # Set image as active
    image = bpy.data.images["Texture"]

    bpy.data.screens['UV Editing'].areas[1].spaces[0].image = image

    # Assign material and texture to the model
    bpy.ops.material.new()
    bpy.ops.object.material_slot_assign()

    bpy.ops.texture.new()

    # Switch to OBJECT mode
    bpy.ops.object.mode_set(mode='OBJECT')


    # Bake vertex colors to a texture
    bpy.context.scene.render.bake_type = 'VERTEX_COLORS'
    bpy.context.scene.render.bake_margin = 16
    bpy.ops.object.bake_image()

    # Save the texture
    output_PNG = argv[1]
    image.filepath_raw = output_PNG
    image.file_format = "PNG"
    image.save()


def unregister():
    pass


if __name__ == "__main__":
    register()