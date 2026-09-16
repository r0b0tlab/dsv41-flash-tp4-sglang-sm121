#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .hermes/closeout/build
exec 9>.hermes/closeout/build/lock
flock -n 9 || { printf 'build owner already active\n'; exit 2; }
[[ ! -e .hermes/closeout/build/ATTEMPTED ]] || { printf 'single rebuild allowance already consumed; inspect receipt\n'; exit 3; }
[[ -z $(git status --porcelain -- adapter docker .dockerignore) ]] || { printf 'uncommitted build inputs\n'; exit 4; }
SHA=$(git rev-parse HEAD)
printf '%s\n' "$SHA" > .hermes/closeout/build/ATTEMPTED
set +e
docker build --progress plain --build-arg "SOURCE_REVISION=$SHA" -f docker/Dockerfile -t dsv41-tp4-sm121:overlay-v2 . > .hermes/closeout/build/build.log 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > .hermes/closeout/build/exit-code
(( rc == 0 )) || exit "$rc"
docker image inspect dsv41-tp4-sm121:overlay-v2 > .hermes/closeout/build/image.json
python3 - "$SHA" <<'PY'
import json,sys,pathlib
p=pathlib.Path('.hermes/closeout/build');d=json.loads((p/'image.json').read_text())[0]
assert d['Architecture']=='arm64'
assert d['Config']['Labels']['org.opencontainers.image.revision']==sys.argv[1]
assert not d['Config'].get('Entrypoint')
(p/'image-id.txt').write_text(d['Id']+'\n')
(p/'PASS').write_text(d['Id']+'\n')
print('BUILD_PASS',d['Id'])
PY
