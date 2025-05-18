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
    "Arctic": (70.0, 190.0, -180.0, 180.0),
    "Southern": (-190.0, -60.0, -180.0, 180.0),
}
EXCEPTIONS = {
    # Only those island groups whose boxes overlap the current OCEAN_BOXES
    "Hawaii": (18.9, 22.3, -160.3, -154.8),
    "Galápagos Islands": (-1.6, 1.667, -92.0167, -89.2667),
    "Pitcairn Islands": (-25.0667, -23.9267, -130.7372, -124.7864),
    "Northern Mariana Islands": (14.9, 18.2, 145.6, 147.1),
    "American Samoa": (-13.35, -11.0, -172.8, -169.19),
    "Bonin Islands": (24.1, 27.1, 142.0, 142.3),
    "Vanuatu": (-20.0, -13.0, 166.0, 171.0),
    "Maldives": (-0.7, 7.2, 72.5, 73.7),
    "British Indian Ocean Territory": (-7.3, -5.4, 71.3, 72.6),
    "Azores": (36.9, 39.7, -31.5, -24.4),
    "Madeira Islands": (32.4, 33.15, -17.3, -16.2),
    "Canary Islands": (27.6, 29.5, -18.3, -13.3),
    "South Georgia and the South Sandwich Islands": (-59.5, -53.0, -38.0, -26.0),
    #"Greenland": (59.0, 83.0, -74.0, -11.0),
    "Svalbard": (76.0, 81.0, 10.0, 35.0),
    #"Japan (main islands & Ogasawara)": (24.0, 46.0, 122.0, 146.0),
    "Cabo Verde": (14.8, 17.2, -25.4, -22.6),
    "Bermuda": (32.2, 32.5, -64.9, -64.5),
    "Seychelles": (-9.7, 4.6, 46.2, 55.4),
    "Mauritius": (-20.8, -19.8, 56.8, 57.9),
    "Réunion":(-21.5, -20.8, 55.0, 55.8),
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
    Snap (lat, lon) out of any ocean box by sampling:
      - 10 points on each edge + 4 corners of its ocean box
      - plus the 10-per-side samples + 4 corners of the three closest exception boxes
    Returns the nearest candidate that's not in any ocean box.
    """
    # helper to get box center
    def center(box):
        y0, y1, x0, x1 = box
        return ((y0+y1)/2, (x0+x1)/2)

    ocean = is_in_ocean(lat, lon)
    if ocean is None:
        return lat, lon

    # gather candidates from the ocean box
    y0, y1, x0, x1 = OCEAN_BOXES[ocean]
    candidates = []
    n_points = 100
    for i in range(n_points):
        t = i / (n_points - 1)
        # bottom, top, left, right edges
        candidates += [
            (y0 - LAT_MARGIN, x0 + t*(x1-x0)),
            (y1 + LAT_MARGIN, x0 + t*(x1-x0)),
            (y0 + t*(y1-y0), x0 - LON_MARGIN),
            (y0 + t*(y1-y0), x1 + LON_MARGIN),
        ]
    # four corners
    candidates += [
        (y0 - LAT_MARGIN, x0 - LON_MARGIN),
        (y0 - LAT_MARGIN, x1 + LON_MARGIN),
        (y1 + LAT_MARGIN, x0 - LON_MARGIN),
        (y1 + LAT_MARGIN, x1 + LON_MARGIN),
    ]

    # now include the three closest exception boxes
    # compute distances to each exception center
    ex_dists = []
    for key, (ey0, ey1, ex0, ex1) in EXCEPTIONS.items():
        cy, cx = center((ey0, ey1, ex0, ex1))
        ex_dists.append(((ey0, ey1, ex0, ex1), (cy-lat)**2 + (cx-lon)**2))
    ex_dists.sort(key=lambda e: e[1])
    n_points = int(n_points /10)
    #print(f"Exception distances: {ex_dists}")
    for box, _ in ex_dists[:3]:
        ey0, ey1, ex0, ex1 = box
        # sample its edges + corners
        
        for i in range(n_points):
            t = i / (n_points - 1)
            candidates += [
                (ey0 + LAT_MARGIN, ex0 + t*(ex1-ex0)), # bottom
                (ey1 - LAT_MARGIN, ex0 + t*(ex1-ex0)), # top
                (ey0 + t*(ey1-ey0), ex0 + LON_MARGIN), # left
                (ey0 + t*(ey1-ey0), ex1 - LON_MARGIN), # right
            ]
        candidates += [
            (ey0 + LAT_MARGIN, ex0 + LON_MARGIN), # bottom left
            (ey0 + LAT_MARGIN, ex1 - LON_MARGIN), # bottom right
            (ey1 - LAT_MARGIN, ex0 + LON_MARGIN), # top left
            (ey1 - LAT_MARGIN, ex1 - LON_MARGIN), # top right
        ]

    # sort by distance from original lat/lon
    candidates.sort(key=lambda c: (c[0]-lat)**2 + (c[1]-lon)**2)
    for ny, nx in candidates:
        if is_in_ocean(ny, nx) is None:
            return ny, nx
    return lat, lon



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
    import random
    # Randomly generate a point in the ocean
    lat = random.uniform(-90, 90)
    lon = random.uniform(-180, 180)
    start = (lat, lon)
    start = (-14,72)
    path = snap_progress(*start)
    
    print("Snapping path:")
    for lat, lon in path:
        print(f"({lat}, {lon})")

