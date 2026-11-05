# SafeGuard

SafeGuard processes webcam or video input to detect possible falls using YOLO
pose estimation.

## Run with Docker

Install Docker with Docker Compose on the machine that will run SafeGuard. From
the project directory, build the image:

```sh
docker compose build
```

The image includes the Python dependencies and the `yolov8n-pose.pt` model
weights. The weights are downloaded during the image build, so building
requires an internet connection.

Put video files in `data/raw/` on the host. Process all supported videos in
that folder:

```sh
docker compose run --rm safeguard
```

Or process one video:

```sh
docker compose run --rm safeguard \
  --headless --source /app/data/raw/your-video.mp4
```

The container runs headlessly and prints telemetry and fall alerts to the
terminal. The Compose service mounts the host's `data/` directory read-only at
`/app/data`.

On Linux, to process the default webcam instead, run:

```sh
docker run --rm --device=/dev/video0:/dev/video0 safeguard:latest
```

The existing `requirements.txt` is used by the Docker build to install the
dependencies; you do not need to install them separately on the other machine.
