#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || $1 != /* || ! -x "$1/initdb" || ! -x "$1/pg_ctl" ]]; then
  printf 'Usage: bash tests/run_egress_postgres.sh /absolute/path/to/postgres/bin\n' >&2
  exit 2
fi
bin=$1
repo=$(cd "$(dirname "$0")/.." && pwd)
python_user_site=$("${PYTHON:-python}" -c 'import site; print(site.getusersitepackages())')
temp=$(mktemp -d /tmp/fbapp-egress-test.XXXXXX)
trap '"$bin/pg_ctl" -D "$temp/db" -m immediate -w stop >/dev/null 2>&1 || :; rm -rf -- "$temp"' EXIT
mkdir -m 700 "$temp/socket" "$temp/home"
export HOME="$temp/home"
unset DATABASE_URL DB_URL PGSERVICE PGHOST PGPORT PGDATABASE PGUSER PGOPTIONS
"$bin/initdb" -D "$temp/db" -U fbapp_egress_test -A trust --encoding=UTF8 --no-locale >/dev/null
"$bin/pg_ctl" -D "$temp/db" -l "$temp/postgres.log" -o "-c listen_addresses='' -k $temp/socket -p 55424" -w start
cd "$repo"
FBAPP_EGRESS_TEST_SOCKET="$temp/socket" PYTHONPATH="$repo:$python_user_site" PYTHONDONTWRITEBYTECODE=1 "${PYTHON:-python}" \
  -m unittest discover -s tests -p 'test_egress_postgres.py' -v
