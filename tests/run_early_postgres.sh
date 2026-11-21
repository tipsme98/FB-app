#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || $1 != /* || ! -x "$1/initdb" || ! -x "$1/pg_ctl" ]]; then
  printf 'Usage: bash tests/run_early_postgres.sh /absolute/path/to/postgres/bin\n' >&2
  exit 2
fi
bin=$1
repo=$(cd "$(dirname "$0")/.." && pwd)
temp=$(mktemp -d /tmp/fbapp-16413-test.XXXXXX)
trap '"$bin/pg_ctl" -D "$temp/db" -m immediate -w stop >/dev/null 2>&1 || :; rm -rf -- "$temp"' EXIT
mkdir -m 700 "$temp/socket" "$temp/home"
export HOME="$temp/home"
unset DATABASE_URL DB_URL EARLY_SETTLE_MODE PGSERVICE PGHOST PGPORT PGDATABASE PGUSER PGOPTIONS
libargs=()
if [[ -f "$bin/../share/postgresql/postgres.bki" ]]; then
  libargs=(-L "$bin/../share/postgresql")
fi
"$bin/initdb" -D "$temp/db" "${libargs[@]}" -U fbapp_early_test -A trust --encoding=UTF8 --no-locale >/dev/null
"$bin/pg_ctl" -D "$temp/db" -l "$temp/postgres.log" -o "-c listen_addresses='' -k $temp/socket -p 55413" -w start
cd "$repo"
FBAPP_TEST_SOCKET="$temp/socket" PYTHONPATH="$repo" "${PYTHON:-python}" -m unittest discover -s tests -p 'test_early_postgres.py' -v
