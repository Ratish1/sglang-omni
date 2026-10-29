until grep -q "queue71 done" /data/ab/queue.log; do
  echo "$(date -u +%T) $(df -m --output=used /dev/shm | tail -1 | tr -d ' ')"
  sleep 2
done > /data/checks/devshm_q70_q71.log
