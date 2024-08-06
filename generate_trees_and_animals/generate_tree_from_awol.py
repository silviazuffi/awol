'''
    Author: Silvia Zuffi
    run as: $BLENDER -b --python generate_tree_from_awol.py
    where $BLENDER is the Blender executable (i.e. /....../Blender.app/Contents/MacOS/Blender)

'''
import bpy
import os
import numpy as np
from math import radians
import sys
import mathutils


file1 = "out_testset_tree_realnvp_mask_tree_clip_features_testset"


render = True

tree_species = ['Ginkgo', 'Coconut', 'Cedar of Lebanon', 'Maritime Pine', 'Fig', 'Cocoa', 'Bigleaf Maple', 'Deodar Cedar', 'Eucalyptus', 'Tulip', 'Oak', 'Banyan', 'American Elm', 'Magnolia', 'Acer', 'Coast Redwood', 'Sequoia', 'Western Red Cedar', 'European Larch', 'Scots Pine', 'White Spruce', 'Italian Cypress']

def rand_in_range(lower, upper):
    return (np.random.random() * (upper - lower)) + lower


def hsv2rgba(hsv):
    # Function from Infinigen
    # Copyright (c) Princeton University.
    # Authors: Alexander Raistrick, Yiming Zuo, Lingjie Mei, Lahav Lipson
    # hsv is a len-3 tuple or array
    c = mathutils.Color()
    c.hsv = list(hsv)
    rgba = list(c) + [1]
    return np.array(rgba)

def hue_round(h):
    if h > 1:
        return (h-1)
    if h < 0:
        return (1+h)
    return h

def get_colors(season='summer'):
    b_hue = 30/360.
    l_hue = 70/360.
    if season == 'summer':
        b_hue = 44/360.
        l_hue = 75/360.
    elif season == 'winter':
        b_hue = 44/360.
        l_hue = 7/360.

    # Sample hue around reference
    b_hue = hue_round(b_hue + np.random.randn()/30.)
    l_hue = hue_round(l_hue + np.random.randn()/30.)

    # Sample S and V in (0,1)
    S = np.random.rand(1)
    V = np.random.rand(1)
    trunk_hsv = (b_hue, S, V)
    branch_hsv = (b_hue, S, V)
    Sl = np.clip(S + np.random.randn(), 0,1)
    Vl = np.clip(V + np.random.randn(), 0,1)
    leaf_hsv = (l_hue, Sl, Vl)
    return hsv2rgba(trunk_hsv), hsv2rgba(branch_hsv), hsv2rgba(leaf_hsv)

def set_tree_params(species, remove_variations=True):
    # First set the default values
    params = {
        'shape': '7',
        'g_scale': 13,
        'g_scale_v': 3,
        'levels': 3,
        'ratio': 0.015,
        'ratio_power': 1.2,
        'flare': 0.6,
        'base_splits': 0,
        'base_size': [0.3, 0.02, 0.02, 0.02],
        'down_angle': [-0, 60, 45, 45],
        'down_angle_v': [-0, -50, 10, 10],
        'rotate': [-0, 140, 140, 77],
        'rotate_v': [-0, 0, 0, 0],
        'branches': [1, 50, 30, 10],
        'length': [1, 0.3, 0.6, 0],
        'length_v': [0, 0, 0, 0],
        'taper': [1, 1, 1, 1],
        'seg_splits': [0, 0, 0, 0],
        'split_angle': [40, 0, 0, 0],
        'split_angle_v': [5, 0, 0, 0],
        'bevel_res': [10, 10, 10, 10],
        'curve_res': [5, 5, 3, 1],
        'curve': [0, -40, -40, 0],
        'curve_back': [0, 0, 0, 0],
        'curve_v': [20, 50, 75, 0],
        'bend_v': [-0, 50, 0, 0],
        'branch_dist': [-0, 0, 0, 0],
        'radius_mod': [1, 1, 1, 1],
        'leaf_blos_num': 40,
        'leaf_shape': '1',
        'leaf_scale': 0.17,
        'leaf_scale_x': 1,
        'leaf_bend': 0.6,
        'blossom_shape': '1',
        'blossom_scale': 0,
        'blossom_rate': 0,
        'tropism': [0, 0, 0.5],
        'prune_ratio': 0,
        'prune_width': 0.5,
        'prune_width_peak': 0.5,
        'prune_power_low': 0.5,
        'prune_power_high': 0.5
    }
    params['seed'] = 11
    if remove_variations:
        params['g_scale_v'] = 0
        params['rotate_v'] = [0, 0, 0, 0]
        params['length_v'] = [0, 0, 0, 0]
        params['split_angle_v'] = [0, 0, 0, 0]
        params['curve_v'] = [0, 0, 0, 0]
        params['bend_v'] = [0, 0, 0, 0]

    return params 

def clip_params(params):
    if params['leaf_shape'] < 1 or params['leaf_shape'] > 10:
        print('unrecognized leaf shape')
        print(params['leaf_shape'])
    params['leaf_shape'] = int(np.round(params['leaf_shape']))
    if params['leaf_shape'] < 1:
        params['leaf_shape'] = 1
    if params['leaf_shape'] > 10:
        params['leaf_shape'] = 10
    params['leaf_shape'] = str(params['leaf_shape'])

    params['leaf_blos_num'] = int(np.round(params['leaf_blos_num']))
    params['leaf_scale'] = np.clip(params['leaf_scale'], 0, 1e2)
    params['leaf_scale_x'] = np.clip(params['leaf_scale_x'], 0, 1e2)
    params['leaf_bend'] = np.clip(params['leaf_bend'], 0, 1e2)
    if params['blossom_shape'] < 1 or params['blossom_shape'] > 3:
        print('unrecognized blossom shape')
        print(params['blossom_shape'])
    params['blossom_shape'] = str(int(np.round(params['blossom_shape'])))
    params['blossom_rate'] = np.clip(params['blossom_rate'], 0, 1e2)
    params['blossom_scale'] = np.clip(params['blossom_scale'], 0, 1e2)
    print(params['shape'])
    if params['shape'] < 0 or params['shape'] > 8:
        print('unrecognized shape')
        print(params['shape'])
    params['shape'] = int(np.round(params['shape']))
    if params['shape'] < 0:
        params['shape'] = 0
    if params['shape'] > 8:
        params['shape'] = 8
    params['shape'] = str(params['shape'])
    params['levels'] = int(np.round(params['levels']))
    params['prune_ratio'] = np.clip(params['prune_ratio'], 0, 1) 
    params['prune_width'] = np.clip(params['prune_width'], 0.000001, 200)
    params['prune_width_peak'] = np.clip(params['prune_width_peak'], 0, 200)
    params['prune_power_low'] = np.clip(params['prune_power_low'], -200, 200)
    params['prune_power_high'] = np.clip(params['prune_power_high'], -200, 200)
    params['base_splits'] = np.clip(int(np.round(params['base_splits'])), -5, 5)
    params['flare'] = np.clip(params['flare'], 0, 10)
    params['g_scale'] = np.clip(params['g_scale'], 0.000001, 150)
    params['g_scale_v'] = np.clip(params['g_scale_v'], 0, 149.99)
    params['tropism'] = [np.clip(b, -10, 10) for b in params['tropism']]
    params['ratio'] = np.clip(params['ratio'], 0.000001, 1)
    params['ratio_power'] = np.clip(params['ratio_power'], 0, 5)
    params['branches'] = [int(np.round(np.clip(b, -500, 500))) for b in params['branches']]
    params['length'] = [np.clip(b, 0, 1) for b in params['length']]
    params['length_v'] = [np.clip(b, 0, 1) for b in params['length_v']]
    params['base_size'] = [np.clip(b, 0, 1) for b in params['base_size']]
    params['branch_dist'] = [np.clip(b, 0, 1) for b in params['branch_dist']]
    params['taper'] = [np.clip(int(np.round(b)), 0, 3) for b in params['taper']]
    params['radius_mod'] = [np.clip(int(np.round(b)), 0, 1) for b in params['radius_mod']]
    params['bevel_res'] = [int(np.round(np.clip(b, 1, 10))) for b in params['bevel_res']]
    params['curve_res'] = [int(np.round(np.clip(b, 1, 10))) for b in params['curve_res']]
    params['curve'] = [int(np.round(np.clip(b, -360, 360))) for b in params['curve']]
    params['curve_v'] = [int(np.round(np.clip(b, -360, 360))) for b in params['curve_v']]
    params['curve_back'] = [int(np.round(np.clip(b, -360, 360))) for b in params['curve_back']]
    params['seg_splits'] = [int(np.round(np.clip(b, 0, 2))) for b in params['seg_splits']]
    params['split_angle'] = [int(np.round(np.clip(b, 0, 360))) for b in params['split_angle']]
    params['split_angle_v'] = [int(np.round(np.clip(b, 0, 360))) for b in params['split_angle_v']]
    params['bend_v'] = [int(np.round(np.clip(b, 0, 360))) for b in params['bend_v']]
    params['down_angle'] = [int(np.round(np.clip(b, 0, 360))) for b in params['down_angle']]
    params['down_angle_v'] = [int(np.round(np.clip(b, -360, 360))) for b in params['down_angle_v']]
    params['rotate'] = [int(np.round(np.clip(b, -360, 360))) for b in params['rotate']]
    params['rotate_v'] = [int(np.round(np.clip(b, 0, 360))) for b in params['rotate_v']]

    return params



def set_custom_tree_params(params):
    # Leaf shape
    leaf_shapes = {'Ovate':'1', 'Linear':'2', 'Cordate':'3', 'Maple':'4', 'Palmate':'5', 'Spiky Oak':'6', 'Rounded Oak':'7', 'Elliptic':'8', 'Rectangle':'9', 'Triangle':'10'}
    bpy.context.scene.tree_leaf_shape_input = params['leaf_shape']

    # Leaf count
    bpy.context.scene.tree_leaf_blos_num_input = params['leaf_blos_num'] #40

    # Leaf scale
    bpy.context.scene.tree_leaf_scale_input = params['leaf_scale'] #0.17

    # Leaf width
    bpy.context.scene.tree_leaf_scale_x_input = params['leaf_scale_x'] #1

    # Leaf bend
    bpy.context.scene.tree_leaf_bend_input = params['leaf_bend'] #0.6

    # Blossom shape
    blossom_shapes = {'Cherry':'1', 'Orange':'2', 'Magnolia':'3'}
    bpy.context.scene.tree_blossom_shape_input = params['blossom_shape'] #blossom_shapes['Magnolia']

    # Blossom rate
    bpy.context.scene.tree_blossom_rate_input = params['blossom_rate'] #0.0

    # Blossom scale
    bpy.context.scene.tree_blossom_scale_input = params['blossom_scale'] #0.1


    # Tree Shape
    tree_shapes = {'Conical', 'Spherical', 'Hemispherical', 'Cylindrical', 'Tapered Cylindrical', 'Flame', 'Inverse Conical', 'Tend Flame', 'Custom'}
    bpy.context.scene.tree_shape_input = params['shape'] #'3'

    # Level Count = number of levels of branching typically 3 or 4
    # Min 1, Max 4
    bpy.context.scene.tree_levels_input = params['levels'] #4

    # Prune Ratio = Fractional amount bt wich the effect of pruning is applied
    # Min 0, Max 1
    bpy.context.scene.tree_prune_ratio_input = params['prune_ratio'] #0

    # Prune Width = Width of the pruning envelope as a fraction of its height (the max heigh of the tree)
    # Min 0.000001, Max 200
    bpy.context.scene.tree_prune_width_input = params['prune_width'] #0.5

    # Prune Width Peak = Tje fractional distance from the bottom of the pruning up at which the peak width occurs
    # Min 0, Max 200
    bpy.context.scene.tree_prune_width_peak_input = params['prune_width_peak'] #0.5

    # Prune Power (low) = The curvature of the lower section of the pruning envelope. < 1 results in a convex shape, > 1 in a concave. 
    # Min -200, Max 200
    bpy.context.scene.tree_prune_power_low_input = params['prune_power_low'] #0.5

    # Prune Power (high) = The curvature of the upper section of the pruning envelope. < 1 results in a convex shape, > 1 in a concave. 
    # Min -200, Max 200
    bpy.context.scene.tree_prune_power_high_input = params['prune_power_high'] #0.5

    # Trunk Splits = Number of splits at base heigh of trunk. If negative, the number of splits will be randomly chosen.
    # Min -5, Max 5
    bpy.context.scene.tree_base_splits_input = params['base_splits']

    # Trunk Flare = How much the radius at the base of the trunk increases.
    # Min 0, Max 10
    bpy.context.scene.tree_flare_input = params['flare'] #0.6

    # Heigh = Scale of the entire tree.
    # Min 0.000001, Max 150
    bpy.context.scene.tree_g_scale_input = params['g_scale'] #13

    # Heigh Variation = Maximum variation in the total size of the tree.
    # Min 0, Max 149.99
    bpy.context.scene.tree_g_scale_v_input = params['g_scale_v'] #3

    # Tropism = Influence upon the growth direction of the tree in the x,y,z directions. The z element only applies to branches in the second level and above. Useful for simulating the effects of gravity, sunlight and wind.
    # Min -10, Max 10
    bpy.context.scene.tree_tropism_input = params['tropism'] # [0, 0, 0.5]

    # Branch thickness Ratio = Ratio of the stem length to radius.
    # Min 0.000001, Max 1
    bpy.context.scene.tree_ratio_input = params['ratio'] #0.015

    # Branch Thickness Ratio Power = How drastically the branch radius is reduced between branching levels.
    # Min 0, Max 5
    bpy.context.scene.tree_ratio_power_input = params['ratio_power'] #1.2

    # Branch Parameters
    # Number = The maximum number of child branches at a given level on each parent branch. The first level parameter indicates the number of trunks coming from the floor, positioned in a rough circle facing outwards (see bamboo). If <0 then all branches are placed in a 'fan' at end of the parent branch 
    # Min -500, Max 500
    bpy.context.scene.tree_branches_input = params['branches'] # [1, 50, 30, 1]

    # Length = The length of branches at a given level as a fraction of their parent branch’s length
    # Min 0, Max 1
    bpy.context.scene.tree_length_input = params['length'] # [1, 0.3, 0.6, 0]

    # Lenght Variation = Maximum variation in branch length
    # Min 0, Max 1
    bpy.context.scene.tree_length_v_input = params['length_v'] # [0, 0, 0, 0]

    # Base Size = Proportion of branch on which no child branches/leaves are spawned
    # Min 0.001
    bpy.context.scene.tree_base_size_input = params['base_size']  # [0.3, 0.02, 0.02, 0.02]

    # Distribution = Controls the distribution of branches along their parent stem. 0 indicates fully alternate branching, interpolating to fully opposite branching at 1. Values > 1 indicate whorled branching (as on fir trees) with n + 1 branches in each whorl. Fractional values result in a rounded integer number of branches in each whorl with rounding error distributed along the trunk
    # Min 0, Max 1
    bpy.context.scene.tree_branch_dist_input = params['branch_dist'] # [0, 0, 0, 0]

    # Taper = Controls the tapering of the radius of each branch along its length. If < 1 then the branch tapers to that fraction of its base radius at its end, so a value 1 results in conical tapering. If =2 the radius remains uniform until the end of the stem where the branch is rounded off in a hemisphere, fractional values between 1 and 2 interpolate between conical tapering and this rounded end. Values > 2 result in periodic tapering with a maximum variation in radius equal to the value − 2 of the base radius - so a value of 3 results in a series of adjacent spheres (see palm trunk)
    # Min 0, Max 3
    bpy.context.scene.tree_taper_input = params['taper'] # [1, 1, 1, 1]

    # Radius Modifier 
    # Min 0, Max 1
    bpy.context.scene.tree_radius_mod_input = params['radius_mod'] # [1, 1, 1, 1]

    # Curve Blevel Resolution = Resolution of curve bevels
    # Min 1, Max 10
    bpy.context.scene.tree_bevel_res_input = params['bevel_res'] # [10, 10, 10, 10]

    # Curve Resolution = Number of segments in each branch
    # Min 1 , Max 10
    bpy.context.scene.tree_curve_res_input =  params['curve_res'] # [5, 5, 3, 1]

    # Curve = Angle by which the direction of the branch will change from start to end, rotating about the branch’s local x-axis
    # Min -360, Max 360
    bpy.context.scene.tree_curve_input = params['curve'] # [0, -40, -40, 0]

    # Curve Variation = Maximum variation in curve angle of a branch. Applied randomly at each segment
    # Min -360, Max 360
    bpy.context.scene.tree_curve_v_input =  params['curve_v'] # [20, 50, 75, 0]

    # Curve Back = Angle in the opposite direction to the curve that the branch will curve back from half way along, creating S shaped branches
    # Min -360, Max 360
    bpy.context.scene.tree_curve_back_input = params['curve_back'] # [0, 0, 0, 0]

    # Segments Splits = Maximum number of dichotomous branches (splits) at each segment of a branch, fractional values are distributed along the branches semi-randomly
    # Min 0, Max 2
    bpy.context.scene.tree_seg_splits_input = params['seg_splits'] # [0, 0, 0, 0]

    # Split Angle = Angle between dichotomous branches
    # Min 0, Max 360
    bpy.context.scene.tree_split_angle_input = params['split_angle'] # [40, 0, 0, 0]

    # Split Angle Variation = Maximum variation in angle between dichotomous branches
    # Min 0, Max 360
    bpy.context.scene.tree_split_angle_v_input = params['split_angle_v'] # [50, 0, 0, 0]

    # Bend Variation  = Maximum angle by which the direction of the branch may change from start to end, rotating about the branch’s local y-axis. Applied randomly at each segment
    # Min 0, Max 360
    bpy.context.scene.tree_bend_v_input = params['bend_v'] # [0, 50, 0, 0]

    # Down Angle = Controls the angle of the direction of a child branch away from that of its parent
    # Min 0, Max 360
    bpy.context.scene.tree_down_angle_input = params['down_angle'] # [0, 60, 45, 45]

    # Down Angle Variation  = Maximum variation in down angle, if < 0 then the value of down angle is distributed along the parent stem
    # Min -360, Max 360
    bpy.context.scene.tree_down_angle_v_input = params['down_angle_v'] # [0, -50, 10, 10]

    # Rotation = Angle around the parent branch between each child branch. If < 0 then child branches are directed this many degrees away from the downward direction in their parent's local basis (see palm leaves). For fanned branches, the fan will spread by this angle and for whorled branches, each whorl will rotate by this angle
    # Min -360, Max 360
    bpy.context.scene.tree_rotate_input =  params['rotate'] # [0, 140, 140, 77]

    # Rotation Variation = Maximum variation in angle between branches. For fanned and whorled branches, each branch will vary in angle by this much
    # Min 0
    bpy.context.scene.tree_rotate_v_input = params['rotate_v'] # [0, 0, 0, 0]


    bpy.context.scene.seed_input = int(params['seed']) 

key_list = ['shape', 'g_scale', 'g_scale_v', 'levels', 'ratio', 'ratio_power', 'flare', 'base_splits', 'base_size', 'down_angle', 'down_angle_v', 'rotate', 'rotate_v', 'branches', 'length', 'length_v', 'taper', 'seg_splits', 'split_angle', 'split_angle_v', 'bevel_res', 'curve_res', 'curve', 'curve_back', 'curve_v', 'bend_v', 'branch_dist', 'radius_mod', 'leaf_blos_num', 'leaf_shape', 'leaf_scale', 'leaf_scale_x', 'leaf_bend', 'blossom_shape', 'blossom_scale', 'blossom_rate', 'tropism', 'prune_ratio', 'prune_width', 'prune_width_peak', 'prune_power_low', 'prune_power_high', 'seed']

num_pred_samples = 1
def main():

    file = file1

    output_image_path = './results/'+ file + '_' + str(num_pred_samples)+'/' 
    output_obj_path = output_image_path

    all_params = np.load(file+'.npy')

    N = int(all_params.shape[0]/num_pred_samples)

    print(N)
    for i in range(N):
        for sample_i in range(num_pred_samples):

            tree_type = tree_species[i] #'test'
            print(tree_type)
            # Cleanup
            for object in bpy.data.objects:
                object.select_set(True)

            bpy.ops.object.delete() # Delete all selected objects

            season = 'summer'

            params = set_tree_params(tree_type, remove_variations=False) #True)
            params_o = params.copy()
            j = 0
            for key in key_list:
                if type(params[key]) is list:
                    k = len(params[key])
                    params[key] = all_params[num_pred_samples*i+sample_i,j:j+k]
                    j = j+k
                else:
                    params[key] = all_params[i,j]
                    j = j+1

            params = clip_params(params)
            set_custom_tree_params(params)

            print("PARAMETERS")
            for key in key_list:
                print(key)
                print(params[key])

            # Save in the network format
            '''
            print(tree_type)
            values = []
            for key in params.keys():
                if not type(params[key]) is list:
                    values.append(float(params[key]))
                else:
                    for v in params[key]:
                        values.append(float(v))
            np.save(tree_type, values)
            '''


            # Create tree
            bpy.ops.object.tree_gen()

            # Convert to mesh
            '''
            bpy.ops.object.tree_gen_convert_to_mesh()

            # Save the tree
            for obj in bpy.context.scene.objects:
                obj.select_set(True)

            name = 'tree'
            fn = os.path.join(output_obj_path, name)

            bpy.ops.wm.obj_export(filepath=fn + '.obj', 
                check_existing=False,
                export_selected_objects=True, 
                export_uv=True, export_normals=True, export_colors=False, export_materials=True, 
                export_triangulated_mesh=True, export_curves_as_nurbs=False, 
                export_object_groups=False, export_material_groups=False, export_vertex_groups=False)

            print("written:", fn)
            '''
        
            bpy.context.scene.render.film_transparent = True

            t_col, b_col, l_col = get_colors(season=season)

            for obj in bpy.context.scene.objects:
                obj.select_set(False)

            for obj in bpy.context.scene.objects:
                if obj.name.startswith('Trunk'):
                    bpy.context.view_layer.objects.active = obj #bpy.context.scene.objects.get("Trunk")
                    bpy.ops.material.new()
                    mat = bpy.data.materials[-1]
                    mat.use_nodes = True
                    mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = t_col #(0.8,0,0,1)
                    obj.data.materials.append(mat)
                    obj.select_set(False)
                if obj.name.startswith('Branches'):
                    bpy.context.view_layer.objects.active = obj #bpy.context.scene.objects.get("Branches")
                    bpy.ops.material.new()
                    mat = bpy.data.materials[-1]
                    mat.use_nodes = True
                    mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = b_col #(0,0,0.8,1)
                    obj.data.materials.append(mat)
                    obj.select_set(False)
                if obj.name.startswith('Leaves'):
                    bpy.context.view_layer.objects.active = obj #bpy.context.scene.objects.get("Branches")
                    bpy.ops.material.new()
                    mat = bpy.data.materials[-1]
                    mat.use_nodes = True
                    mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = l_col #(0,0.8,0,1)
                    obj.data.materials.append(mat)
                    obj.select_set(False)


            # Create camera
            bpy.ops.object.camera_add( location=(0, -40, 6), rotation=(radians(100), 0, 0) )
            camera = bpy.context.selected_objects[0]
            bpy.context.scene.camera = camera # Set new camera as the scene rendering camera

            # Create new light
            lamp_data = bpy.data.lights.new(name="Sun", type='SUN')
            lamp_data.energy = 0.6
            lamp_object = bpy.data.objects.new(name="Sun", object_data=lamp_data)
            bpy.context.collection.objects.link(lamp_object)
            lamp_object.location = (-0.3, 0.44, 1.78)

            lamp_data2 = bpy.data.lights.new(name="Light", type='POINT')
            lamp_data2.energy = 80
            lamp_object2 = bpy.data.objects.new(name="Light", object_data=lamp_data2)
            bpy.context.collection.objects.link(lamp_object2)
            lamp_object2.location = (-1.97, -0.65, 0.77)
            lamp_data3 = bpy.data.lights.new(name="Light", type='POINT')
            lamp_data3.energy = 80
            lamp_object3 = bpy.data.objects.new(name="Light3", object_data=lamp_data3)
            bpy.context.collection.objects.link(lamp_object3)
            lamp_object3.location = (0.86, -0.63, -0.31)

            # Cycles 
            bpy.context.scene.render.engine = 'CYCLES'
            bpy.context.scene.cycles.samples = 128
            bpy.context.scene.cycles.use_denoising = True
            bpy.context.scene.cycles.device = "CPU"        

            # Output
            bpy.context.scene.render.resolution_x = 540 #960 
            bpy.context.scene.render.resolution_y = 960 #540 
            bpy.context.scene.render.resolution_percentage = 100

            #output_image_path_i = output_image_path + 'tmp%0.2d.png' % i 
            output_image_path_i = output_image_path + tree_type+'_'+str(sample_i)+'.png'  
            bpy.context.scene.render.filepath = output_image_path_i
            bpy.ops.render.render(write_still=True)
            print("[INFO] Render finished")


if __name__ == '__main__':
    main()
