#!/usr/bin/env bash
# py-spy run of the stack on card 0 once queue_worst2_c20261010.sh is done.
# usage (in the container): bash pyspy_q_c20261010.sh
until grep -q "queue done" /data/c20261010/queue_worst2.txt 2> /dev/null; do sleep 20; done
bash /data/c20261010/in/pyspy_run_c20261010.sh t_stack 0 0-15,64-79 35000 /data/c20261010/pyspy/stack
