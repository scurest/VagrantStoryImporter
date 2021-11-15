bl_info = {
    "name": "Vagrant Story Model Importer",
    "author": "scurest",
    "version": (1, 0, 0),
    "blender": (2, 93, 0),
    "location": "File > Import",
    "description": "Import Vagrant Story model files",
    "category": "Import",
}

import bpy
from bpy.props import StringProperty, BoolProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Euler

from math import pi
import struct
import os


class ImportVagrantStory(bpy.types.Operator, ImportHelper):
    """Load Vagrant Story model"""
    bl_idname = "import_model.vagrant_story"
    bl_label = "Import Vagrant Story"
    bl_options = {'PRESET', 'UNDO'}

    filter_glob: StringProperty(
        default="*.wep;*.shp;*.zud;*.mpd",
        options={'HIDDEN'},
    )

    set_rest_pose: BoolProperty(
        name="Set Rest Pose",
        description=(
            "Set rest pose to first pose of animation. "
            "Default rest pose is garbage"
        ),
        default=True,
    )

    def execute(self, context):
        import_vs(self, context)
        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout

        layout.label(text="  WEP = Weapon")
        layout.label(text="  SHP = Character")
        layout.label(text="  ZUD = Unit")
        layout.label(text="  MPD = Map")

        layout.separator(factor=4)

        # Get selected filename
        try:
            filename = context.space_data.params.filename.upper()
        except Exception:
            filename = None

        # Show nice name for maps
        if filename in MAP_ZONE_TABLE:
            layout.label(text="Map Name:")
            layout.label(text="  %s" % MAP_ZONE_TABLE[filename][2])


def menu_func_import(self, context):
    self.layout.operator(ImportVagrantStory.bl_idname, text="Vagrant Story")


def register():
    bpy.utils.register_class(ImportVagrantStory)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.utils.unregister_class(ImportVagrantStory)


if __name__ == "__main__":
    register()


###

# Makes Ashley 181 cm tall
SHP_SCALE = 1.81 / 466

# Makes Ashley to scale in MAP009 (Entrance to Darkness)
MPD_SCALE = 2.348 * SHP_SCALE


def import_vs(op, context):
    mdl = load_model(op.filepath)

    is_zud = isinstance(mdl, ZUD)
    if is_zud:
        zud = mdl
        mdl = zud.shp

    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

    if bpy.ops.object.select_all.poll():
        bpy.ops.object.select_all(action='DESELECT')

    new_obs = []

    mesh_ob = create_blender_mesh(mdl)
    new_obs.append(mesh_ob)

    if type(mdl) is SHP:  # no arma for WEPs
        arma_ob = create_blender_arma(mdl)
        new_obs.append(arma_ob)
    else:
        arma_ob = None

    if arma_ob:
        mesh_ob.parent = arma_ob

        mod = mesh_ob.modifiers.new(name="Armature", type='ARMATURE')
        mod.object = arma_ob

        if not is_zud:
            seqs = load_seqs_for_shp(mdl, arma_ob)
        else:
            seqs = load_seqs_for_zud(zud, arma_ob)

        create_animation_data(mdl, arma_ob, seqs)

        if arma_ob.animation_data:
            # Star first track so it will be playing
            arma_ob.animation_data.nla_tracks[-1].is_solo = True

    if is_zud:
        create_zud_weps(zud, arma_ob, new_obs)

    top_ob = arma_ob or mesh_ob
    top_ob.location = context.scene.cursor.location

    if isinstance(mdl, MPD):
        top_ob.scale = [MPD_SCALE] * 3
    else:
        top_ob.scale = [SHP_SCALE] * 3

    view_layer = context.view_layer
    collection = view_layer.active_layer_collection.collection

    for ob in new_obs:
        collection.objects.link(ob)
        ob.select_set(True)

    view_layer.objects.active = top_ob

    if op.set_rest_pose and arma_ob and arma_ob.animation_data:
        # Set pose to first frame of first animation

        ac = arma_ob.animation_data.nla_tracks[-1].strips[0].action
        rot_curves_per_bone = get_rot_curves_per_bone(ac)

        for pbone in arma_ob.pose.bones:
            rot_curves = rot_curves_per_bone.get(pbone.name)

            if not rot_curves: continue

            pbone.rotation_euler[0] = rot_curves[0].evaluate(0)
            pbone.rotation_euler[1] = rot_curves[1].evaluate(0)
            pbone.rotation_euler[2] = rot_curves[2].evaluate(0)

        # Change rest pose to current pose

        set_rest_pose(context)


def load_model(filepath):
    r = Reader.for_file(filepath)
    name = os.path.basename(filepath).upper()
    _, ext = os.path.splitext(filepath)
    ext = ext.upper()

    if ext == ".WEP":
        mdl = WEP(r, name)
    elif ext == ".SHP":
        mdl = SHP(r, name)
    elif ext == ".ZUD":
        mdl = ZUD(r, name)
    elif ext == ".MPD":
        mdl = MPD(r, name)
    else:
        raise RuntimeError("Bad file type: %s" % ext)

    mdl.filepath = filepath

    if ext == ".MPD":
        mdl.znd = load_zone_file_for_map(filepath)

    return mdl


def load_zone_file_for_map(mpd_path):
    mpd_dir = os.path.dirname(mpd_path)
    mpd_name = os.path.basename(mpd_path).upper()

    if mpd_name not in MAP_ZONE_TABLE:
        return None

    zone_num = MAP_ZONE_TABLE[mpd_name][0]
    znd_name = "ZONE%03d.ZND" % zone_num
    znd_path = os.path.join(mpd_dir, znd_name)

    if not os.path.isfile(znd_path):
        # Try lowercase
        znd_path = os.path.join(mpd_dir, znd_name.lower())

    if not os.path.isfile(znd_path):
        return None

    r = Reader.for_file(znd_path)
    name = "ZONE%03d" % zone_num
    return ZND(r, name)


def create_zud_weps(zud, arma_ob, new_obs):
    for i in range(2):
        wep = [zud.weapon, zud.shield][i]
        mount_id = [240, 241][i]

        if not wep: continue

        ob = create_blender_mesh(wep)
        new_obs.append(ob)

        ob.parent = arma_ob

        # Look for a bone we can mount this on
        for bone in zud.shp.bones:
            if bone.mount_id == mount_id:
                bone_name = "Bone.%03d" % bone.id
                ob.parent_type = 'BONE'
                ob.parent_bone = bone_name

                # Move from tail to head of bone
                bone_len = arma_ob.data.bones[bone_name].length
                ob.location[1] = -bone_len

                # Want it to face the right way, but this isn't always
                # right. Maybe bone stores what way it should rotate?
                ob.rotation_euler[2] = pi


def load_seqs_for_shp(wep, arma_ob):
    seqs = find_seq_files(wep.filepath)

    if not seqs:
        # Look for another SHP with the same bones
        # and use its SEQs

        dirname = os.path.dirname(wep.filepath)
        filename = os.path.basename(wep.filepath)
        shp_names = find_shps_with_same_skeleton(filename)

        for shp_name in shp_names:
            shp_path = os.path.join(dirname, shp_name)
            seqs = find_seq_files(shp_path)
            if seqs:
                break

    return seqs


def load_seqs_for_zud(zud, arma_ob):
    # SEQs stored in the ZUD
    seqs = [zud.common_seq, zud.battle_seq]
    seqs = [seq for seq in seqs if seq]

    if not seqs:
        # Look for SEQs for a SHP with the same skeleton
        zud_name = os.path.basename(zud.filepath)
        shp_names = find_shps_with_same_skeleton(zud_name)

        if shp_names:
            # .ZUDs are in root/MAP
            # .SHPs are in root/OBJ
            map_dir = os.path.dirname(zud.filepath)
            obj_dir = os.path.join(map_dir, "..", "OBJ")
            if not os.path.isdir(obj_dir):
                # Try again with lowercase
                obj_dir = os.path.join(map_dir, "..", "obj")

            for shp_name in shp_names:
                shp_path = os.path.join(obj_dir, shp_name)
                shp_seqs = find_seq_files(shp_path)

                if shp_seqs:
                    seqs += shp_seqs
                    break

    return seqs


def create_animation_data(wep, arma_ob, seqs):
    actions = []
    for seq in seqs:
        for anim in seq.animations:
            actions.append(create_seq_action(anim))

    if not actions: return

    if not arma_ob.animation_data:
        arma_ob.animation_data_create()

    tracks = arma_ob.animation_data.nla_tracks

    # Stash actions to NLA tracks
    # Reversed, or else the first one would be on bottom
    for action in reversed(actions):
        new_track = tracks.new(prev=None)
        new_track.name = action.name
        new_track.strips.new(action.name, 1, action)
        new_track.lock = True
        new_track.mute = True


def find_seq_files(shp_path):
    # Given A.SHP, find A*.SEQ files in the same dir
    # Ignores case (ISO filesystem is case insensitive)
    shp_dir = os.path.dirname(shp_path)

    shp_filename = os.path.basename(shp_path)
    shp_prefix, _ = os.path.splitext(shp_filename)
    shp_prefix = shp_prefix.upper()

    try:
        filenames = os.listdir(shp_dir or ".")
    except Exception:
        return []

    seq_names = []

    for filename in filenames:
        filename_upper = filename.upper()
        if filename_upper.startswith(shp_prefix):
            if filename_upper.endswith(".SEQ"):
                seq_names.append(filename)

    seq_names = sort_seq_names(seq_names)

    # Load SEQs
    seqs = []
    for seq_name in seq_names:
        seq_path = os.path.join(shp_dir, seq_name)
        r = Reader.for_file(seq_path)
        name = os.path.basename(seq_path).upper()

        try:
            seq = SEQ(r, name)
            seqs.append(seq)
        except Exception as e:
            print("Error loading %s: %s" % (seq_path, e))

    return seqs


def sort_seq_names(seq_names):
    # Sort .SEQ names, but put COM (common) SEQs first
    # COMs usually have basic walk, stand, etc. actions

    common = []
    others = []

    for name in seq_names:
        if "COM" in name.upper():
            common.append(name)
        else:
            others.append(name)

    common.sort()
    others.sort()

    return common + others


####

# Stuff for creating Blender data


def create_blender_mesh(mdl):
    mesh = bpy.data.meshes.new(mdl.name)
    mesh.from_pydata(mdl.verts, [], mdl.polys)

    layer = mesh.uv_layers.new()
    layer.data.foreach_set("uv", mdl.uvs)

    if mdl.vcolors:
        layer = mesh.vertex_colors.new()
        layer.data.foreach_set("color", mdl.vcolors)

    if hasattr(mdl, "polys_size"):
        layer = mesh.polygon_layers_int.new(name="size")
        layer.data.foreach_set("value", mdl.polys_size)

    if hasattr(mdl, "polys_info"):
        layer = mesh.polygon_layers_int.new(name="info")
        layer.data.foreach_set("value", mdl.polys_info)

    if hasattr(mdl, "polys_type"):
        layer = mesh.polygon_layers_int.new(name="type")
        layer.data.foreach_set("value", mdl.polys_type)

    if hasattr(mdl, "polys_material"):
        mesh.polygons.foreach_set("material_index", mdl.polys_material)

    mats = mdl.build_materials()
    for mat in mats:
        mesh.materials.append(mat)

    mesh.validate()
    mesh.update()

    ob = bpy.data.objects.new(mesh.name, mesh)

    if mdl.groups:
        create_vertex_groups(mdl, ob)

    return ob


def create_vertex_groups(mdl, ob):
    vstart = 0

    for grp in mdl.groups:
        grp_name = "Group.%03d" % grp.id

        if hasattr(grp, "bone_id"):
            if 0 <= grp.bone_id < len(mdl.bones):
                grp_name = "Bone.%03d" % grp.bone_id

        vg = ob.vertex_groups.new(name=grp_name)

        vrange = tuple(range(vstart, grp.last_vertex))
        vg.add(vrange, 1.0, 'REPLACE')
        vstart = grp.last_vertex


def create_blender_arma(wep):
    if not wep.bones:
        return None

    arma = bpy.data.armatures.new("%s Armature" % wep.name)
    ob = bpy.data.objects.new(arma.name, arma)

    # Switch to Edit mode to add bones

    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.context.scene.collection.objects.link(ob)
    orig_active_object = bpy.context.view_layer.objects.active
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode='EDIT')

    create_edit_bones(wep, arma)

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = orig_active_object
    bpy.context.scene.collection.objects.unlink(ob)

    # Set Euler order
    for pbone in ob.pose.bones:
        # VS -> Blender conversion turns XYZ into YXZ (I think)
        pbone.rotation_mode = 'YXZ'

    return ob


def create_edit_bones(wep, arma):
    # First pass: create bones
    for bone in wep.bones:
        ebone = arma.edit_bones.new(name="Bone.%03d" % bone.id)

        ebone.head = (bone.tot_offset, 0, 0)
        ebone.tail =(bone.tot_offset + max(1, -bone.length), 0, 0)
        ebone.roll = -pi/2

        #ebone.use_connect = True  # should we do this?

        ebone["Mount ID"] = bone.mount_id
        ebone["BodyPart ID"] = bone.body_part_id

    # Second pass: set parents
    for bone in wep.bones:
        if bone.has_parent:
            ebone = arma.edit_bones["Bone.%03d" % bone.id]
            parent_ebone = arma.edit_bones["Bone.%03d" % bone.parent_id]
            ebone.parent = parent_ebone


def create_material(name, texture, use_vertex_color=False, use_backface_culling=False):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.blend_method = 'CLIP'
    mat.use_backface_culling = use_backface_culling

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output = nodes["Material Output"]

    nodes.remove(nodes["Principled BSDF"])

    tex_img = nodes.new("ShaderNodeTexImage")
    tex_img.image = texture
    tex_img.interpolation = "Closest"

    # Used for alpha transparency
    mix = nodes.new("ShaderNodeMixShader")
    transp = nodes.new("ShaderNodeBsdfTransparent")

    links.new(output.inputs[0], mix.outputs[0])
    links.new(mix.inputs[1], transp.outputs[0])
    links.new(mix.inputs[0], tex_img.outputs["Alpha"])

    if use_vertex_color:
        # Combines texcolor and vertex color with PS1 formula:
        #   2 * texcolor * vcolor
        mul = nodes.new("ShaderNodeMixRGB")
        mulx2 = nodes.new("ShaderNodeMixRGB")
        vcolor = nodes.new("ShaderNodeVertexColor")

        mul.blend_type = "MULTIPLY"
        mul.inputs[0].default_value = 1.0

        mulx2.label = "Multiply X2"
        mulx2.blend_type = "MULTIPLY"
        mulx2.inputs[0].default_value = 1.0
        # Morally the formula is 2*texcolor*vcolor, so the color 0.5 (0x80)
        # is "fully bright", ie. doesn't change the texture. But the vertex
        # color in Blender's node graph is in linear space, so 0x80 (sRGB)
        # is passed in as 0.216 (linear). So we adjust the factor of 2 to
        # keep 0x80 "fully bright".
        fac = 1 / 0.2158605
        mulx2.inputs[2].default_value = (fac, fac, fac, 1.0)

        links.new(mix.inputs[2], mulx2.outputs[0])
        links.new(mulx2.inputs[1], mul.outputs[0])
        links.new(mul.inputs[1], tex_img.outputs["Color"])
        links.new(mul.inputs[2], vcolor.outputs["Color"])

        mix.location = 110, 120
        transp.location = -90, 30
        mulx2.location = -120, 330
        mul.location = -300, 340
        tex_img.location = -800, 230
        vcolor.location = -500, 100

    else:
        links.new(mix.inputs[2], tex_img.outputs["Color"])

        mix.location = 110, 250
        transp.location = -120, 90
        tex_img.location = -230, 390

    return mat


def create_image(name, w, h, pixels, flip_y=True):
    if flip_y:
        # Texture is upside down, reverse all rows
        new_pixels = []
        row_size = w * 4
        ofs = len(pixels) - row_size

        for _ in range(h):
            new_pixels += pixels[ofs:ofs + row_size]
            ofs -= row_size

        pixels = new_pixels

    img = bpy.data.images.new(name, w, h, alpha=True)
    img.pixels.foreach_set(pixels)
    img.pack()

    return img


def create_seq_action(seq_anim):
    ac = bpy.data.actions.new(seq_anim.name)
    num_bones = seq_anim.seq.num_bones

    for bone_id in range(num_bones):
        bone_name = "Bone.%03d" % bone_id

        # Rotation

        rot_keys = seq_anim.build_rotation_keyframes(bone_id)
        data_path = 'pose.bones["%s"].rotation_euler' % bone_name
        add_fcurves(ac, data_path, bone_name, rot_keys)

        # Scale

        # scale_keys = seq_anim.build_scale_keyframes(bone.id)
        # data_path = 'pose.bones["%s"].scale' % bone_name
        # add_keyframes(ac, data_path, bone.name, scale_keys)

    return ac


def add_fcurves(action, data_path, group_name, keys):
    if not keys: return

    num_keys = len(keys)
    num_components = len(keys[0][1])

    pts = [None] * (2 * num_keys)
    pts[0::2] = (k[0] for k in keys)

    ipo_prop = bpy.types.Keyframe.bl_rna.properties["interpolation"]
    ipo_val = ipo_prop.enum_items['LINEAR'].value   # should be CONSTANT?
    ipos = [ipo_val] * num_keys

    for i in range(num_components):
        fc = action.fcurves.new(
            data_path=data_path,
            index=i,
            action_group=group_name,
        )

        pts[1::2] = (k[1][i] for k in keys)

        fc.keyframe_points.add(num_keys)
        fc.keyframe_points.foreach_set("co", pts)
        fc.keyframe_points.foreach_set("interpolation", ipos)


####

# VS file parsing is below here
# Taken from https://github.com/morris/vstools


class Reader:
    def __init__(self, data, pos=0):
        self.data = data
        self.pos = pos

    @staticmethod
    def for_file(filepath):
        with open(filepath, "rb") as f:
            data = f.read()
        return Reader(data)

    def buffer(self, n):
        buf = self.data[self.pos:self.pos + n]
        self.pos += n
        return buf

    def skip(self, n): self.pos += n

    def unpack(self, fmt):
        fields = struct.unpack_from(fmt, self.data, offset=self.pos)
        self.pos += struct.calcsize(fmt)
        return fields

    def u8(self): return self.unpack("B")[0]
    def s8(self): return self.unpack("b")[0]
    def u16(self): return self.unpack("<H")[0]
    def s16(self): return self.unpack("<h")[0]
    def s16big(self): return self.unpack(">h")[0]
    def u32(self): return self.unpack("<I")[0]
    def s32(self): return self.unpack("<i")[0]


def decode_ps1_colors(colors):
    return [
        # u16 RGB555 -> float[4] RGBA
        (
            ( c & 0x1f ) / 31,
            ( (c>>5) & 0x1f ) / 31,
            ( (c>>10) & 0x1f ) / 31,
            1.0 if c else 0.0,  # black = transparent
        )
        for c in colors
    ]


class WEPBone: pass
class WEPGroup: pass

class WEP:
    # Weapon, also the base class for SHP

    def __init__(self, r, name):
        self.r = r
        self.name = name
        self.read()

    def read(self):
        self.header()
        self.data()

    def header(self):
        r = self.r

        self.header1()

        self.texture_ptr1 = r.u32() + 0x10
        r.skip(0x30)  # padding
        r.texture_ptr = r.u32() + 0x10
        self.group_ptr = r.u32() + 0x10
        self.vertex_ptr = r.u32() + 0x10
        self.face_ptr = r.u32() + 0x10

    def header1(self):
        r = self.r

        magic = r.buffer(4)
        if magic != b"H01\0":
            raise Exception("WEP: wrong magic number: %s" % magic)

        self.num_bones = r.u8()
        self.num_groups = r.u8()

        num_tris = r.u16()       # not actually the number of tris
        num_quads = r.u16()      # not actually the number of quads
        num_remaining = r.u16()
        self.num_polys = num_tris + num_quads + num_remaining

    def data(self):
        self.bone_section()
        self.group_section()
        self.vertex_section()
        self.face_section()
        self.texture_section(num_palettes=7, is_wep=True)

    def bone_section(self):
        r = self.r
        bones = []

        for id in range(self.num_bones):
            bone = WEPBone()
            bone.id = id

            bone.length = r.s32()
            bone.parent_id = r.s8()
            bone.group_id = r.s8()
            bone.mount_id = r.u8()  # for mounting weapons etc.
            bone.body_part_id = r.u8()

            bone.mode = r.s8()  # might affect rotation calculation

            r.skip(7)

            bone.has_parent = (0 <= bone.parent_id < self.num_bones)

            bones.append(bone)

        self.bones = bones

    def group_section(self):
        r = self.r
        groups = []

        for id in range(self.num_groups):
            grp = WEPGroup()
            grp.id = id
            grp.bone_id = r.s16()
            grp.last_vertex = r.u16()

            groups.append(grp)

        self.groups = groups

    def vertex_section(self):
        r = self.r
        verts = []
        num_verts = self.groups[-1].last_vertex

        for  _ in range(num_verts):
            x, y, z = r.unpack("<hhhxx")
            # VS -> Blender space
            verts.append((x, z, -y))

        self.verts = verts
        self.calc_armature_space_verts()

    def face_section(self, colored=False):
        r = self.r
        orig_pos = r.pos

        polys = []
        polys_size = []
        polys_info = []
        uvs = []
        vcolors = []

        for _ in range(self.num_polys):

            if not colored:
                ty = r.u8()

                if not (ty == 0x24 or ty == 0x2c):
                    # Abort and retry as colored faces
                    r.pos = orig_pos
                    return self.face_section(colored=True)

            else:
                ty = r.data[r.pos + 11]

                if not (ty == 0x34 or ty == 0x3c):
                    raise Exception("WEP: unknown poly type: %s" % ty)

            is_quad = (ty == 0x2c or ty == 0x3c)

            if not is_quad:
                if not colored:
                    # Uncolored tri
                    size, info, vert1, vert2, vert3, u1, v1, u2, v2, u3, v3 = \
                        r.unpack("<BBx3H6B")
                else:
                    # Colored tri
                    vert1, vert2, vert3, u1, v1, r1, g1, b1, r2, g2, b2, size, \
                    r3, g3, b3, info, u2, v2, u3, v3 = \
                        r.unpack("<3H5Bx12B")

                polys.append((vert1//4, vert3//4, vert2//4))
                uvs += [u2, v2, u1, v1, u3, v3]
                if colored:
                    vcolors += [
                        r1/255, g1/255, b1/255, 1.0,
                        r3/255, g3/255, b3/255, 1.0,
                        r2/255, g2/255, b2/255, 1.0,
                    ]

            else:
                if not colored:
                    # Uncolored quad
                    size, info, vert1, vert2, vert3, vert4, u1, v1, u2, v2, \
                    u3, v3, u4, v4 = \
                        r.unpack("<BBx4H8B")
                else:
                    # Colored quad
                    vert1, vert2, vert3, vert4, r1, g1, b1, r2, g2, b2, size, \
                    r3, g3, b3, info, r4, g4, b4, u1, v1, u2, v2, u3, v3, u4, v4 = \
                        r.unpack("<4H3Bx11Bx8B")

                polys.append((vert1//4, vert3//4, vert4//4, vert2//4))
                uvs += [u1, v1, u3, v3, u4, v4, u2, v2]
                if colored:
                    vcolors += [
                        r1/255, g1/255, b1/255, 1.0,
                        r3/255, g3/255, b3/255, 1.0,
                        r4/255, g4/255, b4/255, 1.0,
                        r2/255, g2/255, b2/255, 1.0,
                    ]

            polys_size.append(size)
            polys_info.append(info)

        self.polys = polys
        self.polys_size = polys_size
        self.polys_info = polys_info
        self.uvs = uvs
        self.vcolors = vcolors

    def texture_section(self, num_palettes, is_wep=False):
        self.texture_map = WEPTextureMap(self.r, num_palettes, is_wep)

        # Now that we know the texture width/height, we can finish UVs
        self.correct_uvs()

    def correct_uvs(self):
        # Convert UVs from [0,width]x[0,height] -> [0,1]x[0,1]
        # and flip them, since we flip textures
        uvs = self.uvs
        inv_w = 1.0 / self.texture_map.width
        inv_h = 1.0 / self.texture_map.height

        for i in range(0, len(uvs), 2):
            uvs[i] *= inv_w
            uvs[i+1] = 1.0 - inv_h * uvs[i+1]

    def calc_armature_space_verts(self):

        # Calculate bone offsets in armature space
        def calc_total_offset(bone):
            if hasattr(bone, "tot_offset"):
                return

            if not (0 <= bone.parent_id < len(self.bones)):
                bone.tot_offset = 0
            else:
                parent = self.bones[bone.parent_id]
                calc_total_offset(parent)
                bone.tot_offset = parent.tot_offset + (-parent.length)

        for bone in self.bones:
            calc_total_offset(bone)

        # Calculate vert positions in armature space
        new_verts = []
        vstart = 0
        for grp in self.groups:

            if 0 < grp.bone_id <= len(self.bones):
                offset = self.bones[grp.bone_id].tot_offset
                for vi in range(vstart, grp.last_vertex):
                    x, y, z = self.verts[vi]
                    new_verts.append((x + offset, y, z))

            vstart = grp.last_vertex

        self.verts = new_verts

    def build_materials(self):
        img = create_image(
            self.name,
            self.texture_map.width,
            self.texture_map.height,
            self.texture_map.build_pixels(palette_id=0),
        )

        mat = create_material(
            name=self.name,
            texture=img,
            use_vertex_color=bool(self.vcolors),
        )

        return [mat]


class WEPTextureMap:
    def __init__(self, r, num_palettes, is_wep):
        self.r = r
        self.is_wep = is_wep
        self.num_palettes = num_palettes
        self.read()

    def read(self):
        self.header()
        self.palette_section()
        self.texel_section()

    def header(self):
        r = self.r

        self.size = r.u32()

        # version
        # always 1 for WEP
        # SHP and ZUD may have different values
        # determines I4 vs I8 format, apparently
        self.version = r.u8()
        self.bpp = 4 if self.version == 16 else 8

        if self.bpp == 8:
            self.width = r.u8() * 2
        else:
            self.width = r.u8() * 4
        self.height = r.u8() * 2
        self.colors_per_palette = r.u8()

    def palette_section(self):
        palettes = []

        if not self.is_wep:
            for _ in range(self.num_palettes):
                palette = self.read_colors(self.colors_per_palette)
                palettes.append(palette)

        else:
            # first 1/3 of each palette uses common colors
            num_common = self.colors_per_palette // 3
            num_uncommon = self.colors_per_palette - num_common

            common_colors = self.read_colors(num_common)

            for _ in range(self.num_palettes):
                palette = common_colors + self.read_colors(num_uncommon)
                palettes.append(palette)

        # Pad palettes so indexing can't go OOB
        for palette in palettes:
            size = 16 if self.bpp == 4 else 256
            if len(palette) < size:
                pad_amt = size - len(palette)
                palette += [[0.0, 0.0, 0.0, 1.0]] * pad_amt

        self.palettes = palettes

    def read_colors(self, num_colors):
        colors = self.r.unpack("<%dH" % num_colors)
        return decode_ps1_colors(colors)

    def texel_section(self):
        num_rows = self.height
        row_bytes = (self.bpp * self.width) // 8
        self.texel_data = self.r.buffer(row_bytes * num_rows)

    def build_pixels(self, palette_id):
        if self.bpp == 8:
            return self.build_pixels_8bpp(palette_id)
        elif self.bpp == 4:
            return self.build_pixels_4bpp(palette_id)
        else:
            raise Exception("unimplemented")

    def build_pixels_8bpp(self, palette_id):
        pixels = []
        palette = self.palettes[palette_id]

        for b in self.texel_data:
            pixels += palette[b]

        return pixels

    def build_pixels_4bpp(self, palette_id):
        pixels = []
        palette = self.palettes[palette_id]

        for b in self.texel_data:
            pixels += palette[b & 0xf]
            pixels += palette[b >> 4]

        return pixels


class SHP(WEP):
    # Character shape

    def __init__(self, r, name):
        super().__init__(r, name)

    def header(self):
        r = self.r

        self.header1()  # inherited from WEP

        self.overlay_x = []
        self.overlay_y = []
        self.width = []
        self.height = []

        for i in range(8):
            self.overlay_x.append(r.u8())
            self.overlay_y.append(r.u8())
            self.width.append(r.u8())
            self.height.append(r.u8())

        r.skip(0x24)  # unknown
        r.skip(6)     # unknown

        self.menu_pos_y = r.s16()
        r.skip(0xc)   # unknown
        self.shadow_radius = r.s16()
        self.shadow_size_increase = r.s16()
        self.shadow_size_decrease = r.s16()
        r.skip(4)

        self.menu_scale = r.s16()
        r.skip(2)
        self.target_sphere_position = r.s16()
        r.skip(8)

        self.anim_LBAs = r.unpack("<12I")
        self.chain_ids = r.unpack("<12H")
        self.special_LBAs = r.unpack("<4I")

        r.skip(0x20)

        self.magic_ptr = r.u32() + 0xf8
        r.skip(0x30)
        self.akao_ptr = r.u32() + 0xf8
        self.group_ptr = r.u32() + 0xf8
        self.vertex_ptr = r.u32() + 0xf8
        self.face_ptr = r.u32() + 0xf8

    def data(self):
        r = self.r

        # inherited
        self.bone_section()
        self.group_section()
        self.vertex_section()
        self.face_section()

        r.skip(self.magic_ptr - self.akao_ptr)

        # skip magic section
        r.skip(4)
        length = r.u32()
        r.skip(length)

        # inherited
        self.texture_section(num_palettes=2)


class ZUD:
    # ZUD = SHP + WEP(x2) + SEQ(x2)

    def __init__(self, r, name):
        self.r = r
        self.name = name
        self.read()

    def read(self):
        self.header()
        self.data()

    def header(self):
        r = self.r

        self.header_pos = r.pos

        self.id_character = r.u8()
        self.id_weapon = r.u8()
        self.id_weapon_category = r.u8()
        self.id_weapon_material = r.u8()
        self.id_shield = r.u8()
        self.is_shield_material = r.u8()

        r.skip(2)

        self.ptr_character_shp = r.u32()
        self.len_character_shp = r.u32()
        self.ptr_weapon_wep = r.u32()
        self.len_weapon_wep = r.u32()
        self.ptr_shield_wep = r.u32()
        self.len_shield_wep = r.u32()
        self.ptr_common_seq = r.u32()
        self.len_common_seq = r.u32()
        self.ptr_battle_seq = r.u32()
        self.len_battle_seq = r.u32()

    def data(self):
        r = self.r

        self.shp = None
        self.weapon = None
        self.shield = None
        self.common_seq = None
        self.battle_seq = None

        r.pos = self.header_pos
        r.pos += self.ptr_character_shp
        self.shp = SHP(r, self.name)

        r.pos = self.header_pos
        r.pos += self.ptr_weapon_wep
        try:
            self.weapon = WEP(r, "%s Weapon" % self.name)
        except Exception:
            pass

        r.pos = self.header_pos
        r.pos += self.ptr_shield_wep
        try:
            self.shield = WEP(r, "%s Shield" % self.name)
        except Exception:
            pass

        r.pos = self.header_pos
        r.pos += self.ptr_common_seq
        try:
            self.common_seq = SEQ(r, "%s Common" % self.name)
        except Exception:
            pass

        r.pos = self.header_pos
        r.pos += self.ptr_battle_seq
        try:
            self.battle_seq = SEQ(r, "%s Battle" % self.name)
        except Exception:
            pass


class MPDGroup: pass

class MPD:
    # Map, requires ZND for textures

    def __init__(self, r, name):
        self.r = r
        self.name = name
        self.znd = None
        self.read()

    def read(self):
        # We only care about the geometry section
        self.header()
        self.room_header()
        self.room_section()

    def header(self):
        r = self.r

        self.ptr_room_section = r.u32()
        self.len_room_section = r.u32()
        r.skip(40)

    def room_header(self):
        r = self.r

        self.len_geometry_section = r.u32()
        r.skip(23*4)

    def room_section(self):
        self.geometry_section()

    def geometry_section(self):
        self.group_section()
        self.face_section()

    def group_section(self):
        r = self.r

        self.num_groups = r.u32()
        self.groups = []

        for grp_id in range(self.num_groups):
            grp = MPDGroup()
            grp.id = grp_id

            grp.head = r.buffer(64)

            # the header is not well understood
            # it seems that the bits in the second byte are flag bits

            # the following fixes the scaling issues in maps 001 and 002
            if (grp.head[1] & 0x08) > 0:
                grp.scale = 1
            else:
                grp.scale = 8

            self.groups.append(grp)

    def face_section(self):
        r = self.r

        self.verts = []
        self.polys = []
        self.polys_type = []
        self.polys_tex_clut = []
        self.uvs = []
        self.vcolors = []

        for grp in self.groups:
            num_tris = r.u32()
            num_quads = r.u32()

            self.read_faces(grp, num_tris, quads=False)
            self.read_faces(grp, num_quads, quads=True)

            grp.last_vertex = len(self.verts)
            grp.last_poly = len(self.polys)

        self.adjust_uvs()
        self.calculate_material_indices()

    def read_faces(self, grp, num_faces, quads):
        r = self.r
        scale = grp.scale

        for _ in range(num_faces):
            p1x, p1y, p1z, \
            p2dx, p2dy, p2dz, \
            p3dx, p3dy, p3dz, \
            r1, g1, b1, ty, \
            r2, g2, b2, u1, \
            r3, g3, b3, v1, \
            u2, v2, clut_id, \
            u3, v3, texture_id = \
                r.unpack("<3h6b4B10BHBBh")

            # p2-p4 are stored as diffs from p1
            p2x = p1x + scale*p2dx
            p2y = p1y + scale*p2dy
            p2z = p1z + scale*p2dz
            p3x = p1x + scale*p3dx
            p3y = p1y + scale*p3dy
            p3z = p1z + scale*p3dz

            # VS -> Blender space
            p1 = (p1x, p1z, -p1y)
            p2 = (p2x, p2z, -p2y)
            p3 = (p3x, p3z, -p3y)

            vo = len(self.verts)

            if not quads:
                self.verts += [p1, p2, p3]
                self.polys.append((vo, vo+2, vo+1))
                self.uvs += [u2, v2, u1, v1, u3, v3]
                self.vcolors += [
                    r1/255, g1/255, b1/255, 1.0,
                    r3/255, g3/255, b3/255, 1.0,
                    r2/255, g2/255, b2/255, 1.0,
                ]

            else:
                p4dx, p4dy, p4dz, u4, r4, g4, b4, v4 = r.unpack("<bbb5B")

                p4x = p1x + scale*p4dx
                p4y = p1y + scale*p4dy
                p4z = p1z + scale*p4dz

                p4 = (p4x, p4z, -p4y)

                self.verts += [p1, p2, p3, p4]
                self.polys.append((vo, vo+2, vo+3, vo+1))
                self.uvs += [u2, v2, u1, v1, u4, v4, u3, v3]
                self.vcolors += [
                    r1/255, g1/255, b1/255, 1.0,
                    r3/255, g3/255, b3/255, 1.0,
                    r4/255, g4/255, b4/255, 1.0,
                    r2/255, g2/255, b2/255, 1.0,
                ]

            self.polys_type.append(ty)
            self.polys_tex_clut.append((texture_id, clut_id))

    def adjust_uvs(self):
        # All map textures are 256x256
        uvs = self.uvs
        inv_tw = 1/256
        inv_th = 1/256

        for i in range(0, len(uvs), 2):
            uvs[i] *= inv_tw
            uvs[i+1] = 1.0 - uvs[i+1] * inv_th

    def calculate_material_indices(self):
        # Create one slot for each unique (texture_id, clut_id) pair

        # Sort by texture_id so the slot order is nicer
        tex_cluts = list(set(self.polys_tex_clut))
        tex_cluts.sort()

        material_map = {
            tex_clut: index
            for index, tex_clut in enumerate(tex_cluts)
        }

        self.polys_material = [
            material_map[tex_clut]
            for tex_clut in self.polys_tex_clut
        ]
        self.material_list = list(material_map.keys())

    def build_materials(self):
        if self.znd is None:
            # No ZND, use empty material slots
            return [None for _ in self.material_list]

        mats = []

        for texture_id, clut_id in self.material_list:
            name = "%s [%02X,%04X]" % (self.znd.name, texture_id, clut_id)

            # Try to reuse material from previous import
            mat = self.find_material_to_reuse(name, texture_id, clut_id)
            if mat:
                mats.append(mat)
                continue

            img = self.znd.build_image(name, texture_id, clut_id)

            mat = create_material(
                name,
                texture=img,
                use_vertex_color=True,
                use_backface_culling=True,
            )
            mat["Texture ID"] = texture_id
            mat["CLUT ID"] = clut_id

            mats.append(mat)

        return mats

    def find_material_to_reuse(self, name, texture_id, clut_id):
        if name in bpy.data.materials:
            mat = bpy.data.materials[name]

            # Check these too to decrease chance of name collision
            if mat.get("Texture ID") == texture_id:
                if mat.get("CLUT ID") == clut_id:
                    return mat

        return None


class ZND:
    # Zone, contains textures for maps

    def __init__(self, r, name):
        self.r = r
        self.name = name
        self.read()

    def read(self):
        self.header()
        self.data()

    def header(self):
        r = self.r

        self.mpd_ptr = r.u32()
        self.mpd_len = r.u32()
        self.enemy_ptr = r.u32()
        self.enemy_len = r.u32()
        self.tim_ptr = r.u32()
        self.tim_len = r.u32()
        r.skip(8)

    def data(self):
        self.r.skip(self.mpd_len)
        self.r.skip(self.enemy_len)
        self.tim_section()

    def tim_section(self):
        r = self.r
        self.tims = []

        r.skip(16)
        num_tims = r.u32()

        for id in range(num_tims):
            r.skip(4)

            tim = TIM(r)
            tim.id = id

            self.tims.append(tim)

    def build_image(self, img_name, texture_id, clut_id):
        # This is rather optimistic about TIM structure

        clut = self.build_clut(clut_id)
        if not clut: return None

        tim = self.find_tim_for_texture_id(texture_id)
        if not tim: return None

        return tim.build_image(img_name, clut)

    def find_tim_for_texture_id(self, texture_id):
        x = (texture_id * 64) & 0x3ff

        for tim in self.tims:
            if x == tim.org_x:
                return tim

        return None

    def build_clut(self, clut_id):
        x = (clut_id * 16) & 0x3ff
        y = (clut_id * 16) >> 10

        # Find TIM containing the CLUT
        for tim in self.tims:
            inside_x = tim.org_x <= x < tim.org_x + tim.width
            inside_y = tim.org_y <= y < tim.org_y + tim.height

            if inside_x and inside_y:
                return tim.build_clut(x, y)

        return None


class TIM:
    # Basically a raw u16[] to be copied to a rect in VRAM

    def __init__(self, r):
        self.r = r
        self.read()

    def read(self):
        r = self.r

        # 12 byte header
        r.skip(4)  # magic 10 00 00 00
        r.skip(4)  # type flags, always 2
        self.filesize = r.u32()

        self.org_x = r.u16()
        self.org_y = r.u16()
        self.width = r.u16()
        self.height = r.u16()

        num_u16s = self.width * self.height
        self.array = r.unpack("<%dH" % num_u16s)

    def build_clut(self, x, y):
        ox = x - self.org_x
        oy = y - self.org_y
        ofs = oy*self.width + ox

        # 16 colors
        colors = self.array[ofs:ofs + 16]
        return decode_ps1_colors(colors)

    def build_image(self, img_name, clut):
        # 4bpp image
        width = self.width * 4
        height = self.height
        pixels = []

        for n in self.array:
            pixels += clut[n & 0xf]
            pixels += clut[(n >> 4) & 0xf]
            pixels += clut[(n >> 8) & 0xf]
            pixels += clut[(n >> 12) & 0xf]

        return create_image(img_name, width, height, pixels)


class SEQ:
    # Stores bone animations

    def __init__(self, r, name):
        self.r = r
        self.name = name
        self.read()

    def read(self):
        self.header()
        self.data()

    def header(self):
        r = self.r

        # remember base for offsets
        self.base_offset = r.pos

        self.num_slots = r.u16()  # 'slots' is just some random name, purpose unknown
        self.num_bones = r.u8()
        r.skip(1)

        self.size = r.u32()  # file size
        self.data_offset = r.u32() + 8  # offset to animation data
        self.slot_offset = r.u32() + 8  # offset to slots
        self.header_offset = self.slot_offset + self.num_slots  # offset to rotation/key data

    def data(self):
        r = self.r

        # number of animations has to be computed
        # length of all headers / length of one animation header
        self.num_animations = (
            (self.header_offset - self.num_slots - 16) // (4*self.num_bones + 10)
        )

        self.animations = []

        for id in range(self.num_animations):
            anim = SEQAnimation(r, self)
            anim.name = "%s Anim %d" % (self.name, id)

            anim.header()

            self.animations.append(anim)

        # read 'slots'
        # these are animation ids, can be used as in this.animations[id].
        # purpose unknown
        self.slots = r.buffer(self.num_slots)

        # read animation data
        for anim in self.animations:
            anim.data()

    def ptr_data(self, i):
        return self.base_offset + self.header_offset + i


class SEQAnimation:
    def __init__(self, r, seq):
        self.r = r
        self.seq = seq

    def header(self):
        r = self.r

        self.length = r.u16()

        # optional base animation to use initial rotation per bone (pose)
        # -1 means undefined
        self.base_animation_id = r.s8()

        # determines scale key parsing
        self.scale_flags = r.u8()

        # points to special actions per frame, e.g. looping and special effects
        self.ptr_actions = r.u16()

        # points to a translation vector for the animated mesh
        # plus translation keys
        self.ptr_translation = r.u16()

        r.skip(2)

        # pointers to pose and rotation keys for individual bones
        self.ptr_bone_rotation = r.unpack("<%dH" % self.seq.num_bones)

        # pointers to (optional) scale keys for bones
        # only used if scaleFlags & 0x02 is set
        self.ptr_bone_scale = r.unpack("<%dH" % self.seq.num_bones)

    def data(self):
        r = self.r

        # read translation
        r.pos = self.seq.ptr_data(self.ptr_translation)

        self.translation = r.unpack(">hhh")
        self.translation_keys = self.read_keys()

        # Skip actions/special effects
        #if self.ptr_actions > 0:
        #    r.seek(self.seq.ptr_data(self.ptr_actions))
        #    self.read_actions()

        self.rotation_per_bone = []
        self.rotation_keys_per_bone = []
        self.scale_per_bone = []
        self.scale_keys_per_bone = []

        # read bone animation data
        for boneid in range(self.seq.num_bones):
            r.pos = self.seq.ptr_data(self.ptr_bone_rotation[boneid])

            if self.base_animation_id == -1:
                self.rotation_per_bone.append(r.unpack(">hhh"))
            else:
                # else use pose of base animation (at build)
                self.rotation_per_bone.append(None)

            self.rotation_keys_per_bone.append(self.read_keys())

            r.pos = self.seq.ptr_data(self.ptr_bone_scale[boneid])

            if self.scale_flags & 0x1:
                self.scale_per_bone.append(r.unpack("<HHH"))
            else:
                self.scale_per_bone.append(None)

            if self.scale_flags & 0x2:
                self.scale_keys_per_bone.append(self.read_keys())
            else:
                self.scale_keys_per_bone.append(None)

    # read keyframes until 0x00-key is found
    # or animation length is exhausted
    def read_keys(self):
        keys = [(0, 0, 0, 0)]
        f = 0

        while True:
            key = self.read_key()

            if key is None: break

            if key[1] is None: key[1] = keys[-1][1]
            if key[2] is None: key[2] = keys[-1][2]
            if key[3] is None: key[3] = keys[-1][3]

            keys.append(key)

            f += key[0]

            if f >= self.length - 1: break

        return keys

    # read one compressed keyframe into F, X?, Y?, Z? values
    # used for translation, rotation, and scale keys
    # this is basically reverse engineered from 0xafe90 to 0xb0000
    def read_key(self):
        r = self.r

        code = r.u8()

        if code == 0: return None

        f, x, y, z = None, None, None, None

        if (code & 0xe0) > 0:
            # number of frames, byte case

            f = code & 0x1f

            if f == 0x1f:
                f = 0x20 + r.u8()
            else:
                f = 1 + f

        else:
            # number of frames, half word case

            f = code & 0x3

            if f == 0x3:
                f = 4 + r.u8()
            else:
                f = 1 + f

            # half word values

            code = code << 3

            h = r.s16big()

            if (h & 0x4) > 0:
                x = h >> 3
                code = code & 0x60

                if (h & 2) > 0:
                    y = r.s16big()
                    code = code & 0xa0

                if (h & 0x1) > 0:
                    z = r.s16big()
                    code = code & 0xc0
            elif (h & 0x2) > 0:
                y = h >> 3
                code = code & 0xa0

                if (h & 0x1) > 0:
                    z = r.s16big()
                    code = code & 0xc0
            elif (h & 0x1) > 0:
                z = h >> 3
                code = code & 0xc0

        # byte values (fallthrough)

        if (code & 0x80) > 0:
            assert x is None
            x = r.s8()

        if (code & 0x40) > 0:
            assert y is None
            y = r.s8()

        if (code & 0x20) > 0:
            assert z is None
            z = r.s8()

        return [f, x, y, z]

    def build_rotation_keyframes(self, bone_id):
        if self.base_animation_id == -1:
            base_animation = self
        else:
            base_animation = self.seq.animations[self.base_animation_id]

        rot13_to_rad = pi/4096
        base = base_animation.rotation_per_bone[bone_id]
        keys = self.rotation_keys_per_bone[bone_id]
        rx, ry, rz = base
        t = 0
        keyframes = []

        rx *= 2
        ry *= 2
        rz *= 2

        for f, x, y, z in keys:
            t += f
            rx += x * f
            ry += y * f
            rz += z * f

            # 13-bit int to radians
            euler_x = rx * rot13_to_rad
            euler_y = ry * rot13_to_rad
            euler_z = rz * rot13_to_rad

            # VS -> Blender conversion
            euler = (-euler_y, euler_x, euler_z)

            keyframes.append((t, euler))

        return keyframes

    def build_scale_keyframes(self, bone_id):
        # TODO
        pass


####

# Changes the rest pose to the current pose.
# Also fixes up animations for the new rest pose.
#
# Can be called from outside the addon so people can
# change the rest pose, but stil fairly fragile and
# only works on models imported by us.


def set_rest_pose(context):
    arma_ob = context.object
    mesh_ob = None

    if not arma_ob or arma_ob.type != 'ARMATURE':
        return

    # Find mesh child with armature modifier

    for child in arma_ob.children:
        if child.type == 'MESH':
            for mod in child.modifiers:
                if mod.type == 'ARMATURE':
                    mod_name = mod.name
                    mesh_ob = child
                    break

    if not mesh_ob:
        return

    # Prepare pose

    pose_rots = []

    for pbone in arma_ob.pose.bones:
        # Only changing rotation is supported
        pbone.location = 0, 0, 0
        pbone.scale = 1, 1, 1

        pose_rots.append(Euler(
            pbone.rotation_euler,
            pbone.rotation_mode,
        ))

    # Apply current pose to the mesh

    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    context.view_layer.objects.active = mesh_ob

    bpy.ops.object.modifier_apply(modifier=mod_name)

    # Apply current pose to edit bones

    context.view_layer.objects.active = arma_ob

    bpy.ops.object.mode_set(mode='POSE')

    bpy.ops.pose.armature_apply(selected=False)

    bpy.ops.object.mode_set(mode='OBJECT')

    # Find all actions on the armature

    ad = arma_ob.animation_data
    actions = set()

    if arma_ob.animation_data:
        if ad.action:
            actions.add(ad.action)
        for track in ad.nla_tracks:
            for strip in track.strips:
                if strip.action:
                    actions.add(strip.action)

    # Fixup actions

    corrections = [rot.to_matrix().inverted() for rot in pose_rots]

    for ac in actions:
        rot_curves_per_bone = get_rot_curves_per_bone(ac)

        for bone_id, pbone in enumerate(arma_ob.pose.bones):
            # Assume location is not animated
            # Pose scale does not need changing

            # Fixup rotation
            # (btw I just guessed the formula lol)

            rot_curves = rot_curves_per_bone.get(pbone.name)

            if not rot_curves: continue

            num_keys = len(rot_curves[0].keyframe_points)
            co0 = [None] * (2 * num_keys)
            co1 = [None] * (2 * num_keys)
            co2 = [None] * (2 * num_keys)

            rot_curves[0].keyframe_points.foreach_get("co", co0)
            rot_curves[1].keyframe_points.foreach_get("co", co1)
            rot_curves[2].keyframe_points.foreach_get("co", co2)

            for i in range(1, 2*num_keys, 2):
                euler = Euler((co0[i], co1[i], co2[i]), pbone.rotation_mode)

                euler = (
                    (corrections[bone_id] @ euler.to_matrix())
                    .to_euler(pbone.rotation_mode)
                )

                if i > 1:
                    # Make consistent with the last Euler
                    euler.make_compatible(last_euler)

                co0[i] = euler[0]
                co1[i] = euler[1]
                co2[i] = euler[2]

                last_euler = euler

            rot_curves[0].keyframe_points.foreach_set("co", co0)
            rot_curves[1].keyframe_points.foreach_set("co", co1)
            rot_curves[2].keyframe_points.foreach_set("co", co2)

            rot_curves[0].update()
            rot_curves[1].update()
            rot_curves[2].update()

    # Add new Armature modifier

    mod = mesh_ob.modifiers.new(mod_name, type='ARMATURE')
    mod.object = arma_ob


def get_rot_curves_per_bone(action):
    rot_curves_per_bone = {}

    for fc in action.fcurves:
        data_path = fc.data_path

        if not (0 <= fc.array_index < 3): continue

        # Extract NAME from 'pose.bones["NAME"].rotation_euler'
        # (Assume NAME doesn't contain " characters)
        if not data_path.startswith('pose.bones["'): continue
        if not data_path.endswith('"].rotation_euler'): continue
        bone_name = data_path[12:-17]

        if bone_name not in rot_curves_per_bone:
            rot_curves_per_bone[bone_name] = [None, None, None]

        rot_curves_per_bone[bone_name][fc.array_index] = fc

    return rot_curves_per_bone


####

# Below here is tables and stuff


# Groups of models the same skeleton/bone structure,
# ie. the same tuple(bone.length for bone in bones).
# Models in the same group should be able to share SEQs.
SKELETON_TABLE = [
["00.SHP","02.SHP","06.SHP","78.SHP","C9.SHP","CA.SHP","Z027U00.ZUD","Z027U02.ZUD","Z029U00.ZUD","Z056U03.ZUD","Z063U00.ZUD","Z099U00.ZUD"],
["01.SHP","4C.SHP","4E.SHP","AA.SHP","Z001U00.ZUD","Z027U01.ZUD","Z099U02.ZUD","Z210U00.ZUD","Z235U07.ZUD","Z235U08.ZUD"],
["03.SHP","08.SHP","0A.SHP","0B.SHP","9C.SHP","Z049U04.ZUD","Z060U00.ZUD","Z066U00.ZUD"],
["04.SHP","9A.SHP","Z060U02.ZUD"],
["05.SHP","9E.SHP"],
["07.SHP","AD.SHP"],
["09.SHP","Z049U05.ZUD"],
["0C.SHP","0D.SHP","0E.SHP","Z009U02.ZUD","Z009U16.ZUD","Z009U17.ZUD","Z013U15.ZUD","Z055U03.ZUD","Z055U22.ZUD","Z250U09.ZUD"],
["0F.SHP","10.SHP","11.SHP","Z009U03.ZUD","Z009U14.ZUD","Z009U15.ZUD","Z013U21.ZUD","Z015U05.ZUD","Z048U10.ZUD","Z048U11.ZUD","Z048U12.ZUD","Z250U10.ZUD"],
["12.SHP","13.SHP","14.SHP","16.SHP","4B.SHP","54.SHP","80.SHP","92.SHP","93.SHP","94.SHP","95.SHP","A1.SHP","A2.SHP","CB.SHP","CD.SHP","Z009U00.ZUD","Z009U01.ZUD","Z009U06.ZUD","Z009U07.ZUD","Z009U08.ZUD","Z009U09.ZUD","Z009U10.ZUD","Z009U11.ZUD","Z009U12.ZUD","Z009U13.ZUD","Z009U18.ZUD","Z009U19.ZUD","Z009U20.ZUD","Z013U05.ZUD","Z013U06.ZUD","Z013U07.ZUD","Z013U16.ZUD","Z013U17.ZUD","Z013U18.ZUD","Z013U19.ZUD","Z013U20.ZUD","Z013U23.ZUD","Z015U01.ZUD","Z015U02.ZUD","Z024U08.ZUD","Z024U09.ZUD","Z028U02.ZUD","Z028U03.ZUD","Z032U01.ZUD","Z032U02.ZUD","Z032U03.ZUD","Z032U04.ZUD","Z032U05.ZUD","Z032U06.ZUD","Z032U07.ZUD","Z032U08.ZUD","Z032U09.ZUD","Z032U10.ZUD","Z032U11.ZUD","Z032U12.ZUD","Z032U13.ZUD","Z032U14.ZUD","Z032U15.ZUD","Z032U16.ZUD","Z032U17.ZUD","Z032U18.ZUD","Z032U19.ZUD","Z032U20.ZUD","Z032U21.ZUD","Z032U22.ZUD","Z032U23.ZUD","Z032U24.ZUD","Z032U25.ZUD","Z032U26.ZUD","Z032U27.ZUD","Z032U28.ZUD","Z032U29.ZUD","Z040U09.ZUD","Z040U10.ZUD","Z048U17.ZUD","Z048U18.ZUD","Z048U19.ZUD","Z048U20.ZUD","Z057U00.ZUD"],
["15.SHP","Z028U06.ZUD","Z028U07.ZUD","Z048U02.ZUD","Z048U03.ZUD","Z048U04.ZUD","Z048U05.ZUD"],
["17.SHP","19.SHP","Z013U08.ZUD","Z013U09.ZUD","Z013U10.ZUD","Z013U11.ZUD","Z015U03.ZUD","Z015U04.ZUD","Z015U11.ZUD","Z015U12.ZUD","Z015U13.ZUD","Z015U14.ZUD","Z024U03.ZUD","Z024U04.ZUD","Z048U21.ZUD","Z048U22.ZUD","Z048U23.ZUD","Z048U24.ZUD","Z096U01.ZUD","Z250U01.ZUD"],
["18.SHP","Z028U04.ZUD","Z028U05.ZUD","Z048U06.ZUD","Z048U07.ZUD","Z048U08.ZUD","Z048U09.ZUD","Z055U10.ZUD","Z055U11.ZUD","Z055U12.ZUD","Z055U13.ZUD","Z055U14.ZUD","Z055U15.ZUD","Z055U16.ZUD","Z055U17.ZUD","Z055U18.ZUD"],
["1A.SHP","1B.SHP","CE.SHP","Z013U00.ZUD","Z013U01.ZUD","Z013U02.ZUD","Z013U03.ZUD","Z013U04.ZUD","Z015U00.ZUD","Z028U23.ZUD"],
["1C.SHP","Z050U05.ZUD","Z050U06.ZUD","Z050U07.ZUD","Z050U08.ZUD","Z061U02.ZUD","Z096U02.ZUD","Z250U11.ZUD"],
["1D.SHP","Z050U09.ZUD","Z050U10.ZUD"],
["1E.SHP","Z051U03.ZUD","Z051U04.ZUD","Z051U05.ZUD","Z051U06.ZUD","Z051U07.ZUD","Z051U08.ZUD","Z051U09.ZUD","Z061U00.ZUD","Z061U01.ZUD","Z096U03.ZUD","Z099U04.ZUD"],
["1F.SHP","Z051U10.ZUD","Z051U11.ZUD","Z051U12.ZUD","Z051U13.ZUD","Z051U14.ZUD","Z051U15.ZUD","Z051U16.ZUD"],
["20.SHP","Z014U00.ZUD","Z014U01.ZUD","Z015U07.ZUD","Z015U08.ZUD","Z015U16.ZUD","Z028U09.ZUD","Z028U10.ZUD","Z028U11.ZUD","Z028U12.ZUD","Z028U13.ZUD","Z028U14.ZUD","Z028U16.ZUD","Z028U17.ZUD","Z028U18.ZUD","Z028U19.ZUD","Z028U20.ZUD","Z028U21.ZUD","Z051U23.ZUD","Z051U24.ZUD","Z051U25.ZUD","Z051U26.ZUD","Z054U02.ZUD","Z054U03.ZUD","Z096U04.ZUD","Z250U02.ZUD"],
["21.SHP","Z024U07.ZUD","Z048U26.ZUD","Z048U27.ZUD","Z049U00.ZUD","Z049U03.ZUD","Z049U13.ZUD","Z049U14.ZUD","Z056U10.ZUD","Z056U11.ZUD","Z096U05.ZUD","Z097U04.ZUD","Z098U04.ZUD","Z242U00.ZUD","Z250U03.ZUD"],
["22.SHP","Z028U08.ZUD","Z048U16.ZUD","Z055U02.ZUD","Z096U06.ZUD","Z098U02.ZUD","Z242U01.ZUD"],
["23.SHP","Z051U17.ZUD","Z051U18.ZUD","Z051U19.ZUD","Z051U20.ZUD","Z054U04.ZUD","Z054U05.ZUD","Z054U06.ZUD"],
["24.SHP","D0.SHP","Z030U04.ZUD","Z030U05.ZUD","Z030U06.ZUD","Z030U07.ZUD","Z036U00.ZUD","Z053U09.ZUD","Z053U10.ZUD","Z053U11.ZUD","Z053U12.ZUD","Z097U02.ZUD"],
["25.SHP","35.SHP","61.SHP","Z012U00.ZUD","Z012U01.ZUD","Z012U02.ZUD","Z012U03.ZUD","Z030U00.ZUD","Z030U09.ZUD","Z234U11.ZUD"],
["26.SHP","Z050U11.ZUD","Z051U21.ZUD"],
["27.SHP","Z055U06.ZUD","Z055U07.ZUD","Z055U08.ZUD","Z055U09.ZUD"],
["28.SHP","Z011U03.ZUD","Z013U12.ZUD","Z013U22.ZUD","Z015U10.ZUD","Z028U00.ZUD","Z050U04.ZUD","Z242U02.ZUD"],
["29.SHP","2A.SHP","Z009U05.ZUD","Z011U02.ZUD","Z013U13.ZUD","Z015U06.ZUD","Z028U01.ZUD","Z050U01.ZUD","Z096U07.ZUD","Z097U01.ZUD","Z098U01.ZUD","Z234U00.ZUD","Z242U03.ZUD"],
["2B.SHP","Z013U14.ZUD","Z014U02.ZUD","Z015U09.ZUD","Z015U15.ZUD","Z051U22.ZUD","Z051U27.ZUD","Z096U08.ZUD","Z099U03.ZUD","Z234U01.ZUD","Z242U04.ZUD"],
["2C.SHP","Z048U13.ZUD","Z048U14.ZUD","Z048U15.ZUD","Z096U09.ZUD","Z097U05.ZUD","Z098U05.ZUD","Z234U02.ZUD"],
["2D.SHP","Z040U07.ZUD","Z096U10.ZUD","Z234U03.ZUD","Z242U05.ZUD"],
["2E.SHP","Z040U08.ZUD","Z234U04.ZUD"],
["2F.SHP","Z030U08.ZUD","Z051U00.ZUD","Z052U05.ZUD","Z053U13.ZUD","Z234U05.ZUD","Z250U04.ZUD"],
["30.SHP","Z040U06.ZUD","Z050U02.ZUD","Z052U06.ZUD","Z053U14.ZUD","Z097U03.ZUD","Z098U03.ZUD","Z234U06.ZUD","Z250U05.ZUD"],
["31.SHP","Z051U02.ZUD","Z052U07.ZUD","Z053U15.ZUD","Z234U07.ZUD","Z250U06.ZUD"],
["32.SHP","Z030U03.ZUD","Z052U08.ZUD","Z053U00.ZUD","Z053U16.ZUD","Z234U08.ZUD","Z250U07.ZUD"],
["33.SHP","Z028U22.ZUD","Z048U01.ZUD","Z053U20.ZUD","Z234U09.ZUD","Z250U08.ZUD"],
["34.SHP","5F.SHP","60.SHP","Z011U00.ZUD","Z011U04.ZUD","Z030U01.ZUD","Z030U02.ZUD","Z040U02.ZUD","Z048U25.ZUD","Z049U01.ZUD","Z053U03.ZUD","Z053U17.ZUD","Z056U21.ZUD","Z056U22.ZUD","Z056U23.ZUD","Z056U24.ZUD","Z056U25.ZUD","Z056U26.ZUD","Z056U27.ZUD","Z056U28.ZUD","Z056U29.ZUD","Z064U00.ZUD","Z234U10.ZUD"],
["36.SHP","63.SHP","CC.SHP","Z050U03.ZUD","Z050U12.ZUD","Z053U18.ZUD","Z053U19.ZUD","Z056U04.ZUD","Z056U07.ZUD","Z056U17.ZUD","Z056U18.ZUD","Z234U12.ZUD"],
["37.SHP","64.SHP","65.SHP","Z016U00.ZUD","Z054U00.ZUD","Z054U01.ZUD","Z055U00.ZUD","Z234U13.ZUD"],
["38.SHP","66.SHP","67.SHP","Z031U00.ZUD","Z056U00.ZUD","Z056U01.ZUD","Z056U09.ZUD","Z233U00.ZUD","Z233U01.ZUD","Z234U14.ZUD"],
["39.SHP","68.SHP","69.SHP","Z022U00.ZUD","Z041U00.ZUD","Z041U01.ZUD","Z048U00.ZUD","Z233U02.ZUD","Z233U03.ZUD","Z234U15.ZUD"],
["3A.SHP","3B.SHP","6A.SHP","6B.SHP","Z006U00.ZUD","Z050U00.ZUD","Z055U04.ZUD","Z055U05.ZUD","Z234U16.ZUD"],
["3C.SHP","49.SHP","4A.SHP","Z049U02.ZUD","Z049U10.ZUD","Z049U11.ZUD","Z049U12.ZUD","Z234U17.ZUD"],
["3D.SHP","3E.SHP","3F.SHP","40.SHP","41.SHP","42.SHP","43.SHP","Z017U00.ZUD","Z017U01.ZUD","Z024U01.ZUD","Z024U05.ZUD","Z040U00.ZUD","Z040U04.ZUD","Z040U05.ZUD","Z051U01.ZUD","Z053U02.ZUD","Z053U04.ZUD","Z056U02.ZUD","Z064U01.ZUD","Z234U18.ZUD","Z234U19.ZUD","Z235U00.ZUD","Z235U01.ZUD","Z235U02.ZUD","Z235U03.ZUD","Z235U04.ZUD"],
["44.SHP","Z056U06.ZUD","Z235U05.ZUD"],
["47.SHP","48.SHP","Z053U05.ZUD","Z053U06.ZUD","Z053U07.ZUD","Z053U08.ZUD","Z055U01.ZUD","Z055U19.ZUD","Z055U20.ZUD","Z055U21.ZUD"],
["4F.SHP","Z027U03.ZUD","Z235U09.ZUD"],
["51.SHP","82.SHP","9F.SHP","A0.SHP","Z032U00.ZUD","Z060U01.ZUD"],
["52.SHP","71.SHP","72.SHP","81.SHP","85.SHP","86.SHP","87.SHP","88.SHP","89.SHP","8A.SHP","96.SHP","A7.SHP","A8.SHP","C4.SHP","Z001U01.ZUD","Z003U00.ZUD","Z003U01.ZUD","Z040U01.ZUD","Z049U06.ZUD","Z096U00.ZUD","Z097U00.ZUD","Z098U00.ZUD","Z099U01.ZUD","Z210U01.ZUD","Z212U00.ZUD","Z212U01.ZUD","Z250U00.ZUD"],
["58.SHP","Z024U02.ZUD","Z048U28.ZUD","Z048U29.ZUD","Z056U12.ZUD","Z056U13.ZUD"],
["59.SHP","Z048U30.ZUD","Z048U31.ZUD","Z056U05.ZUD","Z056U08.ZUD"],
["5A.SHP","Z024U00.ZUD"],
["5B.SHP","Z022U01.ZUD","Z052U00.ZUD"],
["5C.SHP","Z024U06.ZUD"],
["5D.SHP","Z022U02.ZUD","Z052U01.ZUD"],
["5E.SHP","Z025U00.ZUD"],
["62.SHP","Z053U01.ZUD","Z056U19.ZUD","Z056U20.ZUD"],
["6C.SHP","77.SHP","Z049U07.ZUD","Z049U08.ZUD","Z049U09.ZUD","Z052U02.ZUD","Z052U03.ZUD","Z052U04.ZUD","Z056U14.ZUD","Z056U15.ZUD","Z056U16.ZUD","Z235U06.ZUD"],
["6F.SHP","83.SHP","Z040U03.ZUD","Z063U01.ZUD"],
["7E.SHP","Z009U04.ZUD","Z011U01.ZUD"],
["B2.SHP","C0.SHP"],
["B4.SHP","B6.SHP","BA.SHP","C1.SHP","C2.SHP"],
["B8.SHP","BD.SHP"],
["CF.SHP","Z028U15.ZUD"],
]


def find_shps_with_same_skeleton(filename):
    filename = filename.upper()

    for row in SKELETON_TABLE:
        if filename in row:

            # Take all .SHPs from the front of the row
            for i in range(len(row)):
                if not row[i].endswith(".SHP"):
                    break

            return row[:i]

    return []


# Credit: https://datacrystal.romhacking.net/wiki/Vagrant_Story:rooms_list
MAP_ZONE_TABLE = {
# Wine Cellar
"MAP009.MPD" : (9, 1, "Entrance to Darkness"),
"MAP010.MPD" : (9, 2, "Room of Cheap Red Wine"),
"MAP011.MPD" : (9, 3, "Room of Cheap White Wine"),
"MAP012.MPD" : (9, 4, "Hall of Struggle"),
"MAP013.MPD" : (9, 5, "Smokebarrel Stair"),
"MAP014.MPD" : (9, 6, "Wine Guild Hall"),
"MAP015.MPD" : (9, 7, "Wine Magnate's Chambers"),
"MAP016.MPD" : (9, 8, "Fine Vintage Vault"),
"MAP017.MPD" : (9, 9, "Chamber of Fear"),
"MAP018.MPD" : (9, 10, "The Reckoning Room"),
"MAP019.MPD" : (9, 11, "A Laborer's Thirst"),
"MAP020.MPD" : (9, 12, "The Rich Drown in Wine"),
"MAP021.MPD" : (9, 13, "Room of Rotten Grapes"),
"MAP024.MPD" : (9, 16, "The Greedy One's Den"),
"MAP027.MPD" : (9, 17, "Worker's Breakroom"),
"MAP409.MPD" : (9, 18, "Blackmarket of Wines"),
"MAP025.MPD" : (11, 1, "The Hero's Winehall"),
"MAP026.MPD" : (12, 1, "The Gallows"),
# Catacombs
"MAP028.MPD" : (13, 1, "Hall of Sworn Revenge"),
"MAP029.MPD" : (13, 2, "The Last Blessing"),
"MAP030.MPD" : (13, 3, "The Weeping Corridor"),
"MAP031.MPD" : (13, 4, "Persecution Hall"),
"MAP032.MPD" : (13, 5, "The Lamenting Mother"),
"MAP033.MPD" : (13, 6, "Rodent-Ridden Chamber"),
"MAP034.MPD" : (13, 7, "Shrine to the Martyrs"),
"MAP036.MPD" : (13, 9, "Hall of Dying Hope"),
"MAP037.MPD" : (13, 10, "Bandits' Hideout"),
"MAP038.MPD" : (13, 11, "The Bloody Hallway"),
"MAP039.MPD" : (13, 12, "Faith Overcame Fear"),
"MAP040.MPD" : (13, 13, "The Withered Spring"),
"MAP041.MPD" : (13, 14, "Repent, O ye Sinners"),
"MAP042.MPD" : (13, 15, "The Reaper's Victims"),
"MAP043.MPD" : (13, 16, "The Last Stab of Hope"),
"MAP044.MPD" : (13, 17, "Hallway of Heroes"),
"MAP046.MPD" : (14, 1, "The Beast's Domain"),
"MAP205.MPD" : (42, 1, "Workshop \"Work of Art\""),
# Sanctum
"MAP047.MPD" : (15, 1, "Prisoners' Niche"),
"MAP048.MPD" : (15, 2, "Corridor of the Clerics"),
"MAP049.MPD" : (15, 3, "Priests' Confinement"),
"MAP050.MPD" : (15, 4, "Alchemists' Laboratory"),
"MAP051.MPD" : (15, 5, "Theology Classroom"),
"MAP052.MPD" : (15, 6, "Shrine of the Martyrs"),
"MAP053.MPD" : (15, 7, "Advent Ground"),
"MAP054.MPD" : (15, 8, "Passage of the Refugees"),
"MAP057.MPD" : (15, 11, "Stairway to the Light"),
"MAP058.MPD" : (15, 12, "Hallowed Hope"),
"MAP059.MPD" : (15, 13, "The Academia Corridor"),
"MAP060.MPD" : (16, 1, "Hall of Sacrilege"),
"MAP061.MPD" : (17, 1, "The Cleansing Chantry"),
# Abandoned Mines B1
"MAP260.MPD" : (50, 1, "Dreamers' Entrance"),
"MAP261.MPD" : (50, 2, "Miners' Resting Hall"),
"MAP262.MPD" : (50, 3, "The Crossing"),
"MAP263.MPD" : (50, 4, "Conflict and Accord"),
"MAP264.MPD" : (50, 5, "The Suicide King"),
"MAP265.MPD" : (50, 6, "The End of the Line"),
"MAP266.MPD" : (50, 7, "The Battle's Beginning"),
"MAP267.MPD" : (50, 8, "What Lies Ahead?"),
"MAP268.MPD" : (50, 9, "The Fruits of Friendship"),
"MAP269.MPD" : (50, 10, "The Earthquake's Mark"),
"MAP270.MPD" : (50, 11, "Coal Mine Storage"),
"MAP271.MPD" : (50, 12, "The Passion of Lovers"),
"MAP272.MPD" : (50, 13, "The Hall of Hope"),
"MAP273.MPD" : (50, 14, "The Dark Tunnel"),
"MAP274.MPD" : (50, 15, "Rust in Peace"),
"MAP275.MPD" : (50, 16, "Everwant Passage"),
"MAP276.MPD" : (50, 17, "Mining Regrets"),
"MAP277.MPD" : (50, 18, "The Smeltry"),
"MAP278.MPD" : (50, 19, "Clash of Hyaenas"),
"MAP279.MPD" : (50, 20, "Greed Knows No Bounds"),
"MAP280.MPD" : (50, 21, "Live Long and Prosper"),
"MAP281.MPD" : (50, 22, "Pray to the Mineral Gods"),
"MAP282.MPD" : (50, 23, "Traitor's Parting"),
"MAP283.MPD" : (50, 24, "Escapeway"),
# Abandoned Mines B2
"MAP284.MPD" : (51, 1, "Gambler's Passage"),
"MAP285.MPD" : (51, 2, "Treaty Room"),
"MAP286.MPD" : (51, 3, "The Miner's End"),
"MAP287.MPD" : (51, 4, "Work, Then Die"),
"MAP288.MPD" : (51, 5, "Bandits' Hollow"),
"MAP289.MPD" : (51, 6, "Delusions of Happiness"),
"MAP290.MPD" : (51, 7, "Dining in Darkness"),
"MAP291.MPD" : (51, 8, "Subtellurian Horrors"),
"MAP292.MPD" : (51, 9, "Hidden Resources"),
"MAP293.MPD" : (51, 10, "Way of Lost Children"),
"MAP294.MPD" : (51, 11, "Hall of the Empty Sconce"),
"MAP295.MPD" : (51, 12, "Acolyte's Burial Vault"),
"MAP296.MPD" : (51, 13, "Hall of Contemplation"),
"MAP297.MPD" : (51, 14, "The Abandoned Catspaw"),
"MAP298.MPD" : (51, 15, "Tomb of the Reborn"),
"MAP299.MPD" : (51, 16, "The Fallen Bricklayer"),
"MAP300.MPD" : (51, 17, "Crossing of Blood"),
"MAP301.MPD" : (51, 18, "Fool's Gold, Fool's Loss"),
"MAP302.MPD" : (51, 19, "Cry of the Beast"),
"MAP303.MPD" : (51, 20, "Senses Lost"),
"MAP304.MPD" : (51, 21, "Desire's Passage"),
"MAP305.MPD" : (51, 22, "Kilroy Was Here"),
"MAP306.MPD" : (51, 23, "Suicidal Desires"),
"MAP307.MPD" : (51, 24, "The Ore of Legend"),
"MAP308.MPD" : (51, 25, "Lambs to the Slaughter"),
"MAP309.MPD" : (51, 26, "A Wager of Noble Gold"),
"MAP310.MPD" : (51, 27, "The Lunatic Veins"),
"MAP410.MPD" : (51, 28, "Corridor of Shade"),
"MAP411.MPD" : (51, 29, "Revelation Shaft"),
# Limestone Quarry
"MAP311.MPD" : (53, 1, "Dark Abhors Light"),
"MAP312.MPD" : (53, 2, "Dream of the Holy Land"),
"MAP313.MPD" : (53, 3, "The Ore Road"),
"MAP314.MPD" : (53, 4, "Atone for Eternity"),
"MAP315.MPD" : (53, 5, "The Air Stirs"),
"MAP316.MPD" : (53, 6, "Bonds of Friendship"),
"MAP317.MPD" : (53, 7, "Stair to Sanctuary"),
"MAP318.MPD" : (53, 8, "The Fallen Hall"),
"MAP319.MPD" : (53, 9, "The Rotten Core"),
"MAP320.MPD" : (53, 10, "Bacchus is Cheap"),
"MAP321.MPD" : (53, 11, "Screams of the Wounded"),
"MAP322.MPD" : (53, 12, "The Ore-Bearers"),
"MAP323.MPD" : (53, 13, "The Dreamer's Climb"),
"MAP324.MPD" : (53, 14, "Sinner's Sustenence"),
"MAP325.MPD" : (53, 15, "The Timely Dew of Sleep"),
"MAP326.MPD" : (53, 16, "Companions in Arms"),
"MAP327.MPD" : (53, 17, "The Auction Block"),
"MAP328.MPD" : (53, 18, "Ascension"),
"MAP329.MPD" : (53, 19, "Where the Serpent Hunts"),
"MAP330.MPD" : (53, 20, "Ants Prepare for Winter"),
"MAP331.MPD" : (53, 21, "Drowned in Fleeting Joy"),
"MAP332.MPD" : (53, 22, "The Laborer's Bonfire"),
"MAP333.MPD" : (53, 23, "Stone and Sulfurous Fire"),
"MAP334.MPD" : (53, 24, "Torture Without End"),
"MAP335.MPD" : (53, 25, "Way Down"),
"MAP336.MPD" : (53, 26, "Excavated Hollow"),
"MAP337.MPD" : (53, 27, "Parting Regrets"),
"MAP338.MPD" : (53, 28, "Corridor of Tales"),
"MAP339.MPD" : (53, 29, "Dust Shall Eat the Days"),
"MAP340.MPD" : (53, 30, "Hall of the Wage-Paying"),
"MAP342.MPD" : (53, 32, "Tunnel of the Heartless"),
# Temple of Kiltia
"MAP139.MPD" : (30, 1, "The Dark Coast"),
"MAP140.MPD" : (30, 2, "Hall of Prayer"),
"MAP141.MPD" : (30, 3, "Those who Drink the Dark"),
"MAP142.MPD" : (30, 4, "The Chapel of Meschaunce"),
"MAP143.MPD" : (30, 5, "The Resentful Ones"),
"MAP144.MPD" : (30, 6, "Those who Fear the Light"),
"MAP145.MPD" : (31, 1, "Chamber of Reason"),
"MAP146.MPD" : (31, 2, "Exit to City Center"),
# Great Cathedral B1
"MAP063.MPD" : (22, 1, "Sanity and Madness"),
"MAP067.MPD" : (22, 5, "Truth and Lies"),
"MAP069.MPD" : (22, 7, "Order and Chaos"),
"MAP070.MPD" : (22, 8, "The Victor's Laurels"),
"MAP071.MPD" : (22, 9, "Struggle for the Soul"),
"MAP072.MPD" : (22, 10, "An Offering of Souls"),
# Great Cathedral L1
"MAP079.MPD" : (24, 1, "The Flayed Confessional"),
"MAP080.MPD" : (24, 2, "Monk's Leap"),
"MAP081.MPD" : (24, 3, "Where Darkness Spreads"),
"MAP082.MPD" : (24, 4, "Hieratic Recollections"),
"MAP083.MPD" : (24, 5, "A Light in the Dark"),
"MAP084.MPD" : (24, 6, "The Poisoned Chapel"),
"MAP085.MPD" : (24, 7, "Sin and Punishment"),
"MAP086.MPD" : (24, 8, "Cracked Pleasures"),
"MAP087.MPD" : (24, 9, "Into Holy Battle"),
# Great Cathedral L2
"MAP074.MPD" : (23, 2, "He Screams for Mercy"),
"MAP075.MPD" : (23, 3, "Light and Dark Wage War"),
"MAP076.MPD" : (23, 4, "Abasement from Above"),
"MAP088.MPD" : (24, 10, "Maelstrom of Malice"),
"MAP089.MPD" : (24, 11, "The Acolyte's Weakness"),
"MAP090.MPD" : (24, 12, "The Hall of Broken Vows"),
"MAP091.MPD" : (24, 13, "The Melodics of Madness"),
"MAP092.MPD" : (24, 14, "Free from Base Desires"),
"MAP093.MPD" : (24, 15, "The Convent Room"),
"MAP095.MPD" : (25, 1, "An Arrow into Darkness"),
"MAP096.MPD" : (25, 2, "What Ails You, Kills You"),
# Great Cathedral L3
"MAP077.MPD" : (23, 5, "The Heretics' Story"),
"MAP078.MPD" : (23, 6, "The Wine-Lecher's Fall"),
"MAP094.MPD" : (24, 16, "Hopes of the Idealist"),
"MAP097.MPD" : (25, 3, "Where the Soul Rots"),
"MAP098.MPD" : (25, 4, "Despair of the Fallen"),
# Great Cathedral L4
"MAP099.MPD" : (25, 5, "The Atrium"),
# Forgotten Pathway
"MAP343.MPD" : (54, 1, "Stair to the Sinners"),
"MAP344.MPD" : (54, 2, "Slaugher of the Innocent"),
"MAP345.MPD" : (54, 3, "The Fallen Knight"),
"MAP346.MPD" : (54, 4, "The Oracle Sins No More"),
"MAP347.MPD" : (54, 5, "Awaiting Retribution"),
# Escapeway
"MAP351.MPD" : (52, 1, "Shelter From the Quake"),
"MAP352.MPD" : (52, 2, "Buried Alive"),
"MAP353.MPD" : (52, 3, "Movement of Fear"),
"MAP354.MPD" : (52, 4, "Facing Your Illusions"),
"MAP355.MPD" : (52, 5, "The Darkness Drinks"),
"MAP356.MPD" : (52, 6, "Fear and Loathing"),
"MAP357.MPD" : (52, 7, "Blood and the Beast"),
"MAP358.MPD" : (52, 8, "Where Body and Soul Part"),
# Iron Maiden B1
"MAP359.MPD" : (55, 1, "The Cage"),
"MAP360.MPD" : (55, 2, "The Cauldron"),
"MAP361.MPD" : (55, 3, "Wooden Horse"),
"MAP362.MPD" : (55, 4, "Starvation"),
"MAP363.MPD" : (55, 5, "The Breast Ripper"),
"MAP364.MPD" : (55, 6, "The Pear"),
"MAP365.MPD" : (55, 7, "The Whirligig"),
"MAP366.MPD" : (55, 8, "Spanish Tickler"),
"MAP367.MPD" : (55, 9, "Heretic's Fork"),
"MAP368.MPD" : (55, 10, "The Chair of Spikes"),
"MAP369.MPD" : (55, 11, "Blooding"),
"MAP370.MPD" : (55, 12, "Bootikens"),
"MAP371.MPD" : (55, 13, "Burial"),
"MAP372.MPD" : (55, 14, "Burning"),
"MAP373.MPD" : (55, 15, "Cleansing the Soul"),
"MAP374.MPD" : (55, 16, "The Garotte"),
"MAP375.MPD" : (55, 17, "Hanging"),
"MAP376.MPD" : (55, 18, "Impalement"),
"MAP377.MPD" : (55, 19, "Knotting"),
"MAP378.MPD" : (55, 20, "The Branks"),
"MAP379.MPD" : (55, 21, "The Wheel"),
"MAP380.MPD" : (55, 22, "The Judas Cradle"),
"MAP381.MPD" : (55, 23, "The Ducking Stool"),
# Iron Maiden B2
"MAP382.MPD" : (56, 1, "The Eunics' Lot"),
"MAP383.MPD" : (56, 2, "Ordeal By Fire"),
"MAP384.MPD" : (56, 3, "Tablillas"),
"MAP385.MPD" : (56, 4, "The Oven at Neisse"),
"MAP386.MPD" : (56, 5, "Strangulation"),
"MAP387.MPD" : (56, 6, "Pressing"),
"MAP388.MPD" : (56, 7, "The Strappado"),
"MAP389.MPD" : (56, 8, "The Mind Burns"),
"MAP390.MPD" : (56, 9, "Thumbscrews"),
"MAP391.MPD" : (56, 10, "The Rack"),
"MAP392.MPD" : (56, 11, "The Saw"),
"MAP393.MPD" : (56, 12, "Ordeal By Water"),
"MAP394.MPD" : (56, 13, "The Cold's Bridle"),
"MAP395.MPD" : (56, 14, "Brank"),
"MAP396.MPD" : (56, 15, "The Shin-Vice"),
"MAP397.MPD" : (56, 16, "Squassation"),
"MAP398.MPD" : (56, 17, "The Spider"),
"MAP399.MPD" : (56, 18, "Lead Sprinkler"),
"MAP400.MPD" : (56, 19, "Pendulum"),
"MAP401.MPD" : (56, 20, "Dragging"),
"MAP402.MPD" : (56, 21, "Tongue Slicer"),
"MAP403.MPD" : (56, 22, "Tormentum Insomniae"),
# Iron Maiden B3
"MAP404.MPD" : (56, 23, "The Iron Maiden"),
"MAP405.MPD" : (56, 24, "Saint Elmo's Belt"),
"MAP406.MPD" : (56, 25, "Judgement"),
"MAP407.MPD" : (56, 26, "Dunking the Witch"),
# Undercity West
"MAP210.MPD" : (47, 1, "Workshop \"Godhands\""),
"MAP220.MPD" : (48, 1, "The Bread Peddler's Way"),
"MAP221.MPD" : (48, 2, "Way of the Mother Lode"),
"MAP222.MPD" : (48, 3, "Sewer of Ravenous Rats"),
"MAP223.MPD" : (48, 4, "Underdark Fishmarket"),
"MAP224.MPD" : (48, 5, "The Sunless Way"),
"MAP225.MPD" : (48, 6, "Remembering Days of Yore"),
"MAP226.MPD" : (48, 7, "Where the Hunter Climbed"),
"MAP227.MPD" : (48, 8, "Larder for a Lean Winter"),
"MAP228.MPD" : (48, 9, "Hall of Poverty"),
"MAP229.MPD" : (48, 10, "The Washing-Woman's Way"),
"MAP230.MPD" : (48, 11, "Beggars of the Mouthharp"),
"MAP231.MPD" : (48, 12, "Corner of the Wretched"),
"MAP232.MPD" : (48, 13, "Path to the Greengrocer"),
"MAP233.MPD" : (48, 14, "Crossroads of Rest"),
"MAP234.MPD" : (48, 15, "Path of the Children"),
"MAP235.MPD" : (48, 16, "Fear of the Fall"),
"MAP236.MPD" : (48, 17, "Sinner's Corner"),
"MAP237.MPD" : (48, 18, "Nameless Dark Oblivion"),
"MAP238.MPD" : (48, 19, "Corner of Prayers"),
"MAP239.MPD" : (48, 20, "Hope Obstructed"),
"MAP240.MPD" : (48, 21, "The Children's Hideout"),
"MAP241.MPD" : (48, 22, "The Crumbling Market"),
"MAP242.MPD" : (48, 23, "Tears from Empty Sockets"),
"MAP243.MPD" : (48, 24, "Where Flood Waters Ran"),
"MAP244.MPD" : (48, 25, "The Body Fragile Yields"),
"MAP245.MPD" : (48, 26, "Salvation for the Mother"),
"MAP246.MPD" : (48, 27, "Bite the Master's Wounds"),
# Undercity East
"MAP247.MPD" : (49, 1, "Hall to a New World"),
"MAP248.MPD" : (49, 2, "Place of Free Words"),
"MAP249.MPD" : (49, 3, "Bazaar of the Bizarre"),
"MAP250.MPD" : (49, 4, "Noble Gold and Silk"),
"MAP251.MPD" : (49, 5, "A Knight Sells his Sword"),
"MAP252.MPD" : (49, 6, "Gemsword Blackmarket"),
"MAP253.MPD" : (49, 7, "The Pirate's Son"),
"MAP254.MPD" : (49, 8, "Sale of the Sword"),
"MAP255.MPD" : (49, 9, "Weapons Not Allowed"),
"MAP256.MPD" : (49, 10, "The Greengrocer's Stair"),
"MAP257.MPD" : (49, 11, "Where Black Waters Ran"),
"MAP258.MPD" : (49, 12, "Arms Against Invaders"),
"MAP259.MPD" : (49, 13, "Catspaw Blackmarket"),
# The Keep
"MAP124.MPD" : (29, 1, "The Warrior's Rest"),
"MAP126.MPD" : (29, 3, "The Soldier's Bedding"),
"MAP127.MPD" : (29, 4, "A Storm of Arrows"),
"MAP128.MPD" : (29, 5, "Urge the Boy On"),
"MAP129.MPD" : (29, 6, "A Taste of the Spoils"),
"MAP130.MPD" : (29, 7, "Wiping Blood from Blades"),
"MAP207.MPD" : (44, 1, "Wkshop \"Keane's Crafts\""),
# City Walls West
"MAP105.MPD" : (28, 1, "Students of Death"),
"MAP106.MPD" : (28, 2, "The Gabled Hall"),
"MAP107.MPD" : (28, 3, "Where the Master Fell"),
# City Walls South
"MAP108.MPD" : (28, 4, "The Weeping Boy"),
"MAP109.MPD" : (28, 5, "Swords for the Land"),
"MAP110.MPD" : (28, 6, "In Wait of the Foe"),
"MAP111.MPD" : (28, 7, "Where Weary Riders Rest"),
"MAP112.MPD" : (28, 8, "The Boy's Training Room"),
# City Walls East
"MAP113.MPD" : (28, 9, "Train and Grow Strong"),
"MAP114.MPD" : (28, 10, "The Squire's Gathering"),
"MAP115.MPD" : (28, 11, "The Invaders are Found"),
"MAP116.MPD" : (28, 12, "The Dream-Weavers"),
"MAP117.MPD" : (28, 13, "The Cornered Savage"),
# City Walls North
"MAP118.MPD" : (28, 14, "Traces of Invasion Past"),
"MAP119.MPD" : (28, 15, "From Squire to Knight"),
"MAP120.MPD" : (28, 16, "Be for Battle Prepared"),
"MAP121.MPD" : (28, 17, "Destruction and Rebirth"),
"MAP122.MPD" : (28, 18, "From Boy to Hero"),
"MAP123.MPD" : (28, 19, "A Welcome Invasion"),
# Snowfly Forest
"MAP179.MPD" : (40, 1, "The Hunt Begins"),
"MAP180.MPD" : (40, 2, "Which Way Home"),
"MAP181.MPD" : (40, 3, "The Giving Trees"),
"MAP182.MPD" : (40, 4, "The Wounded Boar"),
"MAP183.MPD" : (40, 5, "Golden Egg Way"),
"MAP184.MPD" : (40, 6, "The Birds and the Bees"),
"MAP185.MPD" : (40, 7, "The Woodcutter's Run"),
"MAP186.MPD" : (40, 8, "The Wolves' Choice"),
"MAP187.MPD" : (40, 9, "Howl of the Wolf King"),
"MAP188.MPD" : (40, 10, "Fluttering Hope"),
"MAP189.MPD" : (40, 11, "Traces of the Beast"),
"MAP190.MPD" : (40, 12, "The Yellow Wood"),
"MAP191.MPD" : (40, 13, "They Also Feed"),
"MAP192.MPD" : (40, 14, "Where Soft Rains Fell"),
"MAP193.MPD" : (40, 15, "The Spirit Trees"),
"MAP194.MPD" : (40, 16, "The Silent Hedges"),
"MAP195.MPD" : (40, 17, "Lamenting to the Moon"),
"MAP196.MPD" : (40, 18, "The Hollow Hills"),
"MAP197.MPD" : (40, 19, "Running with the Wolves"),
"MAP198.MPD" : (40, 20, "You Are the Prey"),
"MAP199.MPD" : (40, 21, "The Secret Path"),
"MAP200.MPD" : (40, 22, "The Faerie Circle"),
"MAP201.MPD" : (40, 23, "Return to the Land"),
"MAP202.MPD" : (40, 24, "Forest River"),
"MAP203.MPD" : (40, 25, "Hewn from Nature"),
"MAP204.MPD" : (40, 26, "The Wood Gate"),
# Snowfly Forest East
"MAP348.MPD" : (41, 1, "Steady the Boar-Spears"),
"MAP349.MPD" : (41, 2, "The Boar's Revenge"),
"MAP350.MPD" : (41, 3, "Nature's Womb"),
# Town Center West
"MAP147.MPD" : (32, 1, "Rue Vermillion"),
"MAP148.MPD" : (32, 2, "The Rene Coastroad"),
"MAP149.MPD" : (32, 3, "Rue Mal Fallde"),
"MAP150.MPD" : (32, 4, "Tircolas Flow"),
"MAP151.MPD" : (32, 5, "Glacialdra Kirk Ruins"),
"MAP152.MPD" : (32, 6, "Rue Bouquet"),
"MAP153.MPD" : (32, 7, "Villeport Way"),
"MAP154.MPD" : (32, 8, "Rue Sant D'alsa"),
"MAP173.MPD" : (34, 1, "Dinas Walk"),
"MAP206.MPD" : (43, 1, "Workshop \"Magic Hammer\""),
# Town Center South
"MAP155.MPD" : (32, 9, "Valdiman Gates"),
"MAP156.MPD" : (32, 10, "Rue Faltes"),
"MAP157.MPD" : (32, 11, "Forcas Rise"),
"MAP158.MPD" : (32, 12, "Rue Aliano"),
"MAP159.MPD" : (32, 13, "Rue Volnac"),
"MAP160.MPD" : (32, 14, "Rue Morgue"),
"MAP174.MPD" : (35, 1, "Zebel's Walk"),
"MAP176.MPD" : (37, 1, "The House Khazabas"),
# Town Center East
"MAP162.MPD" : (32, 16, "Rue Lejour"),
"MAP163.MPD" : (32, 17, "Kesch Bridge"),
"MAP164.MPD" : (32, 18, "Rue Crimnade"),
"MAP165.MPD" : (32, 19, "Rue Fisserano"),
"MAP166.MPD" : (32, 20, "Shasras Hill Park"),
"MAP175.MPD" : (36, 1, "Gharmes Walk"),
"MAP177.MPD" : (38, 1, "The House Gilgitte"),
"MAP171.MPD" : (39, 1, "Plateia Lumitar"),
"MAP208.MPD" : (45, 1, "Workshop \"Metal Works\""),
"MAP209.MPD" : (46, 1, "Wkshop \"Junction Point\""),
# Miscellaneous
"MAP022.MPD" : (9, 14, "Wine Cellar: Chamber of Fear (after the quake)"),
"MAP023.MPD" : (9, 15, "Ashley and Merlose at the Wine Cellar gate"),
"MAP412.MPD" : (9, 19, "Wine Cellar: Room of Rotten Grapes (after defeating Lich)"),
"MAP211.MPD" : (10, 1, "Ashley and Merlose outside the Wine Cellar gate"),
"MAP408.MPD" : (12, 1, "Wine Cellar: The Gallows (Mino Zombie, new chest)"),
"MAP035.MPD" : (13, 8, "Catacombs: unused room"),
"MAP045.MPD" : (13, 18, "Catacombs: The Lamenting Mother (after the quake)"),
"MAP055.MPD" : (15, 9, "Sanctum: Passage of the Refugees (after cloudstone activated)"),
"MAP056.MPD" : (15, 10, "Sanctum: unused room"),
"MAP062.MPD" : (18, 1, "Bardorba and Rosencrantz"),
"MAP212.MPD" : (19, 1, "Ashley's flashback"),
"MAP213.MPD" : (20, 1, "VKP briefing"),
"MAP214.MPD" : (21, 1, "Ashley meets Merlose outside manor"),
"MAP064.MPD" : (22, 2, "Great Cathedral: unused room"),
"MAP065.MPD" : (22, 3, "Great Cathedral: unused room"),
"MAP066.MPD" : (22, 4, "Great Cathedral: unused room"),
"MAP068.MPD" : (22, 6, "Great Cathedral: unused room"),
"MAP073.MPD" : (23, 1, "Great Cathedral: unused room"),
"MAP100.MPD" : (26, 1, "Ashley finds Sydney in the Cathedral"),
"MAP101.MPD" : (27, 1, "Guildestern's \"ascension\""),
"MAP102.MPD" : (27, 2, "Ashley fights Guildestern, part 2"),
"MAP125.MPD" : (29, 2, "Keep: unused room"),
"MAP161.MPD" : (32, 15, "Town Center: unused room"),
"MAP167.MPD" : (32, 21, "Town Center: unused room"),
"MAP168.MPD" : (32, 22, "Town Center: unused room"),
"MAP169.MPD" : (32, 23, "Town Center: unused room"),
"MAP170.MPD" : (32, 24, "Town Center: unused room"),
"MAP172.MPD" : (33, 1, "Merlose finds corpses at Leà Monde's entrance"),
"MAP341.MPD" : (53, 31, "Limestone Quarry: unused room"),
"MAP103.MPD" : (57, 1, "Great Cathedral: Hardin's past"),
"MAP104.MPD" : (58, 1, "Joshua's dream"),
"MAP413.MPD" : (59, 1, "Great Cathedral: Sydney reaches Hardin and Guildestern"),
"MAP131.MPD" : (60, 1, "Escape from Leà Monde"),
"MAP132.MPD" : (61, 1, "Collapse of Leà Monde"),
"MAP133.MPD" : (62, 1, "Collapse of Leà Monde"),
"MAP134.MPD" : (63, 1, "Collapse of Leà Monde"),
"MAP135.MPD" : (64, 1, "Collapse of Leà Monde"),
"MAP136.MPD" : (65, 1, "Big Bang"),
"MAP137.MPD" : (66, 1, "Hardin dies"),
"MAP138.MPD" : (67, 1, "Bardorba reunion"),
"MAP178.MPD" : (68, 1, "So began the story of the vagrant"),
"MAP414.MPD" : (69, 1, "Merlose questions Hardin"),
"MAP415.MPD" : (70, 1, "The Dark tempts Ashley"),
"MAP427.MPD" : (96, 1, "Debug pathfinding"),
"MAP428.MPD" : (97, 1, "Debug"),
"MAP429.MPD" : (98, 1, "Debug"),
"MAP430.MPD" : (99, 1, "Debug AI"),
"MAP000.MPD" : (100, 1, "Debug traps, goto degub room"),
"MAP506.MPD" : (250, 1, "Debug room (MAPs)"),
}
