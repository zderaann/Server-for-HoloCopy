bl_info = {
    "name": "Decimate and Bake",
    "author": "AZe",
    "version": (1, 0),
    "blender": (2, 75, 0),
    "warning": "",
    "wiki_url": "",
    "category": "Add Mesh",
}

import bpy
from bpy.types import Operator
from bpy.props import FloatVectorProperty
from bpy_extras.object_utils import AddObjectHelper, object_data_add
from mathutils import Vector
import sys


def register():

    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    bpy.ops.import_mesh.ply(filepath=argv[0] + "/meshed-poisson.ply")

    # Decimation
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            modDec = obj.modifiers.new("Decimate", type="DECIMATE")
            bpy.context.scene.update()
            # len(obj.data.vertices)

            while modDec.face_count >= 20000:
                modDec.ratio *= 0.5
                bpy.context.scene.update()

    bpy.ops.object.modifier_apply(modifier="Decimate")

    bpy.ops.image.new(name="Texture", width=4096, height=4096, color=(0.0, 0.0, 0.0, 0.0), alpha=True,
                      generated_type='BLANK', float=False, gen_context='NONE', use_stereo_3d=False)

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.0, user_area_weight=0.0, use_aspect=True,
                             stretch_to_bounds=True)
    bpy.ops.uv.select_all(action='SELECT')

    # Set image as active
    image = bpy.data.images["Texture"]

    bpy.data.screens['UV Editing'].areas[1].spaces[0].image = image

    # Assign material and texture to decimated model
    bpy.ops.material.new()
    bpy.ops.object.material_slot_assign()

    bpy.ops.texture.new()

    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.export_mesh.ply(filepath=argv[0] + "/decimated.ply", use_normals=False, use_uv_coords=True)

    # Texture baking










    bpy.ops.import_mesh.ply(filepath=argv[0] + "\\meshed-poisson.ply")

    # Create image, unwrap decimated model
    bpy.ops.object.select_all(action='DESELECT')



    # Select models and set decimated to active

    bpy.ops.object.select_all(action='DESELECT')

    bpy.data.objects['meshed-poisson.001'].select = True

    bpy.data.objects['meshed-poisson'].select = True
    bpy.context.scene.objects.active = bpy.data.objects['meshed-poisson']

    # Bake vertex colors to a texture
    bpy.context.scene.render.bake_type = 'VERTEX_COLORS'
    bpy.context.scene.render.bake_margin = 16
    bpy.context.scene.render.use_bake_selected_to_active = True
    bpy.ops.object.bake_image()

    # Save the texture
    image.filepath_raw = argv[0] + "\\texture.png"
    image.file_format = "PNG"
    image.save()



def unregister():
    pass


if __name__ == "__main__":
    register()