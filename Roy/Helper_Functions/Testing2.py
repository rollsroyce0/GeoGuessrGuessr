import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Define ocean bounding boxes and land exceptions
OCEAN_BOXES = {
    # Pacific split into four strictly-ocean quadrants (no major landmasses)
    "Pacific_NW": (0.0, 41.8, 142.08, 180.0),
    "Pacific_NW2": (41.8, 51.0, 157.0, 180.0),
    "Pacific_NE": (15.0, 48.6, -180.0, -124.7),
    "Pacific_NE2": (0.0, 15.0, -180.0, -93.0),
    "Pacific_SE": (-60.0, 0.0, -180.0, -82.0),

    # Atlantic split into precise ocean-only regions
    "Atlantic_NW": (20.0, 41.0, -71.5, -59.0),
    "Atlantic_NW2": (8.5, 46.5, -59.0, -30.0),
    "Atlantic_NW3": (46.5, 60.0, -52.0, -30.0),
    "Atlantic_NE": (29.2, 63.0, -30.0, -10.5),
    "Atlantic_NE2": (4.25, 29.2, -30.0, -18.0),
    "Atlantic_NE3": (0.0, 4.25, -30.0, 0.0),
    "Atlantic_SW": (-35.2, 0.0, -34.5, -10.0),
    "Atlantic_SE": (-35.2, 0.0, -10.0, 8.7),
    "Atlantic_S2": (-60.0, -35.2, -56.0, 136.0),

    # Indian Ocean trimmed to avoid land
    "Indian_N": (-30.0, 5.8, 51.0, 95.0),
    "Indian_S": (-35.2, -30.0, 32.0, 114.0),

    # Polar seas
    "Arctic": (70.0, 90.0, -180.0, 180.0),
    "Southern": (-90.0, -60.0, -180.0, 180.0),
}
EXCEPTIONS = {
    # Land exceptions to keep (e.g., islands)
    "Hawaii": (18.9, 22.25, -160.25, -154.8),
}

# Margins in degrees
LON_MARGIN = 0.001
LAT_MARGIN = 0.001


def is_in_ocean(lat, lon):
    """
    Return the ocean_key if (lat, lon) is within a defined ocean box,
    excluding any exception rectangles (land islands).
    """
    # normalize lon to [-180,180]
    if lon > 180: lon -= 360
    if lon < -180: lon += 360

    for ocean, (y0, y1, x0, x1) in OCEAN_BOXES.items():
        # longitude wrap
        if x0 < x1:
            in_lon = x0 < lon < x1
        else:
            in_lon = lon > x0 or lon < x1
        if y0 < lat < y1 and in_lon:
            # check exceptions
            for ex, (ey0, ey1, ex0, ex1) in EXCEPTIONS.items():
                if ex0 < ex1:
                    in_ex_lon = ex0 < lon < ex1
                else:
                    in_ex_lon = lon > ex0 or lon < ex1
                if ey0 < lat < ey1 and in_ex_lon:
                    break
            else:
                return ocean
    return None


def snap_point(lat, lon, depth=0):
    """
    Find the closest point outside the ocean box by sampling
    10 points on each edge plus four corners, and return it directly.
    """
    ocean = is_in_ocean(lat, lon)
    if ocean is None:
        return lat, lon
    y0, y1, x0, x1 = OCEAN_BOXES[ocean]
    candidates = []
    # sample 10 points per edge
    n_points = 3
    for i in range(n_points):
        t = i / (n_points)
        # bottom, top, left, right
        candidates += [
            (y0 - LAT_MARGIN, x0 + t*(x1-x0)+LON_MARGIN), # bottom
            (y1 + LAT_MARGIN, x0 + t*(x1-x0)+LON_MARGIN), # top
            (y0 + t*(y1-y0)+LAT_MARGIN, x0 - LON_MARGIN), # left
            (y0 + t*(y1-y0)+LAT_MARGIN, x1 + LON_MARGIN), # right
        ]
    # corners
    candidates += [
        (y0 - LAT_MARGIN, x0 - LON_MARGIN), # bottom left
        (y0 - LAT_MARGIN, x1 + LON_MARGIN), # bottom right
        (y1 + LAT_MARGIN, x0 - LON_MARGIN), # top left
        (y1 + LAT_MARGIN, x1 + LON_MARGIN), # top right
    ]
    # choose nearest
    #print(f"Snapping depth {depth} for ({lat}, {lon})")
    print(f"Candidates: {candidates}")
    # sort the candidates by distance
    candidates.sort(key=lambda c: (c[0]-lat)**2 + (c[1]-lon)**2)
    # find the first candidate that is not in ocean
    for c in candidates:
        if is_in_ocean(c[0], c[1]) is None:
            new_lat, new_lon = c
            break
    
    # plot all the candidates
    fig = plt.figure(figsize=(14,7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    # draw boxes
    for y0, y1, x0, x1 in OCEAN_BOXES.values():
        rect = patches.Rectangle((x0, y0), x1-x0, y1-y0,
                                 transform=ccrs.PlateCarree(), alpha=0.3)
        ax.add_patch(rect)
    # draw exceptions
    for ey0, ey1, ex0, ex1 in EXCEPTIONS.values():
        rect = patches.Rectangle((ex0, ey0), ex1-ex0, ey1-ey0,
                                 transform=ccrs.PlateCarree(), edgecolor='red', facecolor='red', alpha=0.3)
        ax.add_patch(rect)
    # plot snapping path
    ax.plot(lon, lat, 'ro', transform=ccrs.PlateCarree())
    ax.plot(new_lon, new_lat, 'go', transform=ccrs.PlateCarree())
    for (olat, olon), (nlat, nlon) in zip(candidates, candidates[1:]):
        nlat, nlon = lat, lon
        ax.plot(olon, olat, 'ro', transform=ccrs.PlateCarree())
        ax.arrow(olon, olat, nlon-olon, nlat-olat,
                 transform=ccrs.PlateCarree(), head_width=1, length_includes_head=True)
    # start and end markers
    ax.plot(lon, lat, 'go', transform=ccrs.PlateCarree(), label='Start')
    ax.plot(new_lon, new_lat, 'k*', transform=ccrs.PlateCarree(), label='End')
    ax.legend()
    plt.title('Ocean Snapper')
    plt.show()
    return new_lat, new_lon


def snap_progress(lat, lon, depth2=0):
    """
    Return the sequence of (lat, lon) during snapping for visualization.
    """
    
    
    path = [(lat, lon)]
    new_lat, new_lon = lat, lon
    while True:
        depth2 += 1
        print(f"Snapping depth2 {depth2} for ({lat}, {lon})")
        ocean = is_in_ocean(new_lat, new_lon)
        if ocean is None:
            break
        new_lat, new_lon = snap_point(new_lat, new_lon)
        path.append((new_lat, new_lon))
    return path


if __name__ == '__main__':
    # Example usage: visualize progress
    start = (-35, -20)  # Example coordinates
    path = snap_progress(*start)
    
    print("Snapping path:")
    for lat, lon in path:
        print(f"({lat}, {lon})")

    fig = plt.figure(figsize=(14,7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    # draw boxes
    for y0, y1, x0, x1 in OCEAN_BOXES.values():
        rect = patches.Rectangle((x0, y0), x1-x0, y1-y0,
                                 transform=ccrs.PlateCarree(), alpha=0.3)
        ax.add_patch(rect)
    # draw exceptions
    for ey0, ey1, ex0, ex1 in EXCEPTIONS.values():
        rect = patches.Rectangle((ex0, ey0), ex1-ex0, ey1-ey0,
                                 transform=ccrs.PlateCarree(), edgecolor='red', facecolor='red', alpha=0.3)
        ax.add_patch(rect)
    # plot snapping path
    for (olat, olon), (nlat, nlon) in zip(path, path[1:]):
        ax.plot(olon, olat, 'ro', transform=ccrs.PlateCarree())
        ax.arrow(olon, olat, nlon-olon, nlat-olat,
                 transform=ccrs.PlateCarree(), head_width=1, length_includes_head=True)
    # start and end markers
    ax.plot(start[1], start[0], 'go', transform=ccrs.PlateCarree(), label='Start')
    ax.plot(path[-1][1], path[-1][0], 'k*', transform=ccrs.PlateCarree(), label='End')
    ax.legend()
    plt.title('Ocean Snapper Progress')
    plt.show()