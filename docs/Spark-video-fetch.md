# Spark video fetch (yt-dlp)

A small service on the Spark that downloads a public video (YouTube and the
sites yt-dlp reads; no DRM) at 480p so a link can become a style reference
of a bible without uploading a file. Files: `docker/spark-stack/video-fetch/`
(`app.py`, `Dockerfile`, `requirements.txt`, `compose.yaml`).

```
Maestro controller ── POST /api/fetch {url} ──▶ video-fetch (Spark, :8095)
                   ◀─ GET /api/fetch/{id} (status, progress) ──┘
                   ◀─ GET /api/fetch/{id}/file (the 480p MP4) ─┘
                      then the usual reference analysis (480p proxy, analyst)
```

## Setup (Dockge stack `~/stacks/video-fetch`)

```bash
# On the Spark
mkdir -p ~/stacks/video-fetch && cd ~/stacks/video-fetch
cp ~/stacks/Maestro/docker/spark-stack/video-fetch/compose.yaml compose.yml
printf 'VIDEO_FETCH_TOKEN=%s\n' "$(openssl rand -hex 24)" > .env   # never print it
mkdir -p data && docker compose up -d --build
curl -s http://127.0.0.1:8095/healthz
```

`MAESTRO_REPO_DIR` (default `../Maestro`) is the repository checkout the image
builds from; `git pull` it before `--build`. tsdproxy publishes the service as
`sp1-video-fetch` on the tailnet, like llama-swap and the model hub.

| Variable | Default | Meaning |
|---|---|---|
| `VIDEO_FETCH_TOKEN` | required | bearer token of every `/api` call |
| `TTL_HOURS` | 48 | finished jobs and their files are removed after this |
| `ALLOWED_HOSTS` | empty (any public host) | optional allow-list of hosts (subdomains included) |
| `UID_GID` | 1000:1000 | owner of `./data` |

## In Maestro

Settings → Services: `director_video_fetch_url`
(`https://sp1-video-fetch.<tailnet>.ts.net`, env `MAESTRO_VIDEO_FETCH_URL`)
and `director_video_fetch_api_key` (the token, env
`MAESTRO_VIDEO_FETCH_API_KEY`, masked like every key).

Bible panel → Style references → paste links, one per line (a label applies
only to a single link; the others take the video's title), scene tags and
the graphics mode as for an upload, "Fetch N links and analyse": the links
queue, two download at a time, every file is analysed in turn. The controller queues a fetch
(`POST /api/v1/director/bibles/{bible}/style-references/from-url`,
`GET .../fetches` lists the recent ones), mirrors the download progress,
streams the file into the bible, registers the reference (label defaults to
the video title) and starts the analysis. The remote job is deleted once the
file is in the bible.

## API

| Call | Result |
|---|---|
| `POST /api/fetch` `{"url": "...", "max_height": 480}` | `{"job": {...}}` status `queued` |
| `GET /api/fetch` | `{"jobs": [...]}` |
| `GET /api/fetch/{id}` | `{"job": {"status": "queued|downloading|done|failed", "progress", "title", "duration_sec", "bytes", "filename", "error", ...}}` |
| `GET /api/fetch/{id}/file` | the MP4 |
| `DELETE /api/fetch/{id}` | removes the job and its folder |
| `GET /healthz` | no token |

A URL must resolve to public addresses: loopback, private (10/8, 172.16/12,
192.168/16), link-local and reserved ranges, and names that do not resolve
(docker service names) are refused with HTTP 400, so the service cannot be
used to reach this host's network. (A redirect met during the download is
not re-checked; only the token holder can submit a URL.)

yt-dlp picks the best streams at or under `max_height`, H.264 and AAC when
available, merged into MP4 by ffmpeg. One download at a time (`PARALLEL`).
Node is the JavaScript runtime yt-dlp needs for YouTube; `yt-dlp-ejs` ships
the challenge scripts so nothing is fetched from GitHub at run time. Update
the image (`docker compose build --pull && docker compose up -d`) when
YouTube changes break downloads: yt-dlp releases follow within days.
