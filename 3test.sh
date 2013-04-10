#!/bin/bash
# Should run in ~15 minutes for 100 iterations
# TA Success rate: 2 / 200 (~1.0%)
echo " " > trials/test3.trials
NUMTRIALS=100
echo " "

for i in `seq 1 $NUMTRIALS`
do
  echo "Running trial $i..."
  python wumpus.py NaiveSafe -R 60 -W 3 -P 7 -B 7 -A 5 -M 0.5 -S1 0.80 -S2 0.70 -S3 0.20 -S4 0.15 >> trials/test3.trials
  trap "echo '\nExited test!'; exit;" SIGINT SIGTERM
done

echo "Successes out of $NUMTRIALS:" >> results/test3.results
grep -c "SCREAM" trials/test3.trials >> results/test3.results
echo "\nSuccesses out of $NUMTRIALS:"
echo `grep -c "SCREAM" trials/test3.trials`
echo "\nTest completed!"
