# Step 0b: simulator image — STOPPED before the rebuild (local disk too full)

Written 2026-11-17 05:20 IST, shakaal. Everything below was read-only: no image was pulled or built, no drive was run, and nobody's images or containers were touched.

## Stop reason

The rebuild needs about 78 GB on the root disk, and the root disk has 21 GB free.

- **Docker storage:** `/var/lib/docker`, on `/dev/nvme1n1p2`, which is the root disk. It is 1.8 T, 99 % used, with **21 GB free** (`df -h /`). Nothing would go to the NAS.
- **Sizes of the last builds of these images:**
  - `alpasim-base:0.134.0`: about 35.8 GB on disk (the 2026-10-08 build);
  - `nre-ga:26.04`: about 42.5 GB on disk (14.3 GB compressed);
  - together about 78 GB, minus any shared layers.
- **Why I stopped instead of launching:** on 2026-11-12 the build was not a separate step; the first check drive's `docker compose` pulled and built both images (`logs/cl12_queue.log:4824-4828`).
  - The run scripts only require 4 GiB free on root (`s20_run_*_scene.sh`, `free_gib`), so a check drive would start the pull and fill the root disk for every user on shakaal.
  - `{bulk_disk}` (37 GB free) is not docker's storage, so it does not help.

## Why the images are gone (read-only findings)

**What can be seen:**
- **Both of our images are gone**, the renderer `nre-ga:26.04` included.
- **So are our two keep-placeholder containers** (`siddesh-keep-alpasim-base`, `siddesh-keep-nre-ga`), which held the images in place. In fact the whole machine now has **0 stopped containers** (`docker ps -a --filter status=exited` returns none).
- **It was not a blanket prune of unused images.** 17 unused images of other users survive, including ones older than our 2026-11-12 build (for example `billy-diariser:nemotron` from 2026-11-03 and `foamalert-dashboard:latest` from 2026-11-02). So whoever removed images either picked ours or picked the largest ones.

**What cannot be seen:**
- **No docker restart or reboot since 2026-11-06,** and the daemon's in-memory event buffer reaches back only a few hours (only exec events).
- **The journal and logs are closed to me.** `journalctl -u docker` shows nothing for this user, and `/var/lib/docker` is not readable.
- **So I cannot tell who removed them, or when** (sometime after the CL18 drives on 2026-11-14).

**A guess, not verified:** with the root disk at 99 %, someone freeing space removed all stopped containers and then the large images that no running container used. Ours were about 78 GB.

## Options (your call)

1. **Ask an admin or the person who freed space** for about 80 GB on the root disk, or for docker's storage to be moved. Then rebuild as pre-registered.
2. **Free root-disk space of our own.**
   - `{home}` holds 50 GB, but I haven't looked at what is safe to remove and won't delete anything without your go-ahead.
   - The 18.8 GB docker build cache is shared, so it isn't ours to prune.
3. **Once rebuilt, protect the images.** Recreate the keep-placeholder containers and tell the team that `siddesh-keep-*` containers hold ~78 GB of images on purpose. This is the third loss: 2026-10-08, 2026-11-03 and now.

## Prepared and ready (no runs)

- `reports/CL20_PREREG.md`: the image check, written before any check drive.
  - It has the 6 drives, their reference runs and the PASS rule (all 6 match on score and on bit-identical per-tick metrics, else STOP).
  - It also records the known risk that the CL12 carry run was unpinned.
- **Still to write before the check:**
  - the check queue: a copy of `s12_queue.sh` part 1 with the data8t root, the carry drives added and no image-id gate for these 6 drives;
  - the comparison script: `s12_imagecheck_cmp.py` extended to the two carry drives.
- **Step 1 onward:** I don't have that part of the brief yet.
