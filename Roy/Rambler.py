import requests
from PIL import Image
import re
import json
import io
import math

MAPS_PREVIEW_ID = "CAEIBAgFCAYgAQ"
UNKNOWN_PREVIEW_CONSTANT = 45.12133303837374
latitude = 52.09
longitude = 5.12
zoom = 3

client_id = re.search(
    '"],null,0,"[^"]+"', requests.get(url="https://www.google.com/maps").json()
).group()[11:-1]

preview_document = json.loads(
    requests.get(
        url="https://www.google.com/maps/preview/photo?authuser=0&hl=en&gl=us&pb=!1e3!5m54!2m2!1i203!2i100!3m3!2i4!3s%s!5b1!7m42!1m3!1e1!2b0!3e3!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e8!2b0!3e3!1m3!1e10!2b0!3e3!1m3!1e10!2b1!3e2!1m3!1e9!2b1!3e2!1m3!1e10!2b0!3e3!1m3!1e10!2b1!3e2!1m3!1e10!2b0!3e4!2b1!4b1!8m0!9b0!11m1!4b1!6m3!1s%s!7e81!15i11021!9m2!2d%f!3d%f!10d%f"
        % (MAPS_PREVIEW_ID, client_id, longitude, latitude, UNKNOWN_PREVIEW_CONSTANT)
    ).text[4:]
)
sphere_id = preview_document[0][0][0]

photometa_document = json.loads(
    requests.get(
        url="https://www.google.com/maps/photometa/v1?authuser=0&hl=en&gl=us&pb=!1m4!1smaps_sv.tactile!11m2!2m1!1b1!2m2!1sen!2sus!3m3!1m2!1e2!2s%s!4m57!1e1!1e2!1e3!1e4!1e5!1e6!1e8!1e12!2m1!1e1!4m1!1i48!5m1!1e1!5m1!1e2!6m1!1e1!6m1!1e2!9m36!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e3!2b1!3e2!1m3!1e3!2b0!3e3!1m3!1e8!2b0!3e3!1m3!1e1!2b0!3e3!1m3!1e4!2b0!3e3!1m3!1e10!2b1!3e2!1m3!1e10!2b0!3e3"
        % (sphere_id)
    ).text[4:]
)

width = int(photometa_document[1][0][2][2][0] / pow(2, 4 - zoom))
height = int(photometa_document[1][0][2][2][0] / pow(2, 5 - zoom))
tiles_width = math.ceil(width / 512)
tiles_height = math.ceil(height / 512)

image_output = Image.new(mode="RGB", size=(width, height))
for x in range(tiles_width):
    for y in range(tiles_height):
        image_chunk = Image.open(
            io.BytesIO(
                requests.get(
                    url="https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid=%s&x=%d&y=%d&zoom=%d&nbt=1&fover=2"
                    % (sphere_id, x, y, zoom)
                ).content
            )
        )
        image_output.paste(image_chunk, (x * 512, y * 512))

image_output.save("photo-sphere.png")