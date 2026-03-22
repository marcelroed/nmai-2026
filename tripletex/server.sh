#!/usr/bin/env bash

#!/usr/bin/env bash

LOG_LEVEL=DEBUG uv run src/tripletex/main.py &
server_pid=$!

sleep 0.5
./make-test-request.py 2>&1 | sed -u $'s/.*/\\x1b[34m&\\x1b[0m/'
request_status=$?

kill "$server_pid"
wait "$server_pid" 2>/dev/null

exit "$request_status"
