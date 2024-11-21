# from https://notebook.community/neuromusic/neuronexus-probe-data/getting%20the%20geometries
# https://github.com/neuromusic/neuronexus-probe-data

from cdcp.paths import PROJECT_DIR
import pandas as pd
import numpy as np
probe_dataframe = pd.read_csv(PROJECT_DIR / 'cdcp' / 'probe_geom'/"NiPOD-ProbeSpec-denormalized.csv")

from scipy import spatial
def get_graph_from_geometry(geometry):
    # let's transform the geometry into lists of channel names and coordinates
    chans,coords = zip(*[(ch,xy) for ch,xy in geometry.items()])
    # we'll perform the triangulation and extract the
    try:
        tri = spatial.Delaunay(coords)
    except:
        x,y = zip(*coords)
        coords = list(coords)
        coords.append((max(x)+1,max(y)+1))
        tri = spatial.Delaunay(coords)
    # then build the list of edges from the triangulation
    indices, indptr = tri.vertex_neighbor_vertices
    edges = []
    for k in range(indices.shape[0]-1):
        for j in indptr[indices[k]:indices[k+1]]:
            try:
                edges.append((chans[k],chans[j]))
            except IndexError:
                # ignore dummy site
                pass
    return edges

def build_geometries(channel_groups):
    for gr, group in channel_groups.items():
        group['graph'] = get_graph_from_geometry(group['geometry'])
    return channel_groups

def get_probe_channel_groups(probe):
    if probe.DesignType=='Linear':
        channel_groups = {}
        for shank in range(int(probe.NumShank)):
            channel_groups[shank] = {}
            sites = shank*int(probe.NumSitePerShank) + np.arange(int(probe.NumSitePerShank))
            x_locs = [0.0 for s in sites]
            y_locs = [probe.TrueSiteSpacing * s for s in sites]
            channel_groups[shank]['channels'] = list(sites)
            channel_groups[shank]['geometry'] = {s:(x,y) for s,x,y in zip(sites,x_locs,y_locs)}
    if probe.DesignType == 'Tetrode':
        channel_groups = {}
        for shank in range(int(probe.NumShank)):
            channel_groups[shank] = {}
            sites = shank*int(probe.NumSitePerShank) + np.arange(int(probe.NumSitePerShank))
            x_locs = [0.0 for s in sites]
            y_locs = [probe.TrueSiteSpacing * s for s in sites]
            channel_groups[shank]['channels'] = list(sites)
            channel_groups[shank]['geometry'] = {s:(x,y) for s,x,y in zip(sites,x_locs,y_locs)}
    return build_geometries(channel_groups)