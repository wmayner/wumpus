#!/bin/bash
# Should run in ~30 seconds for 100 iterations
# TA Success rate: 84 / 1,000 (~0.8%)
echo " " > trials/test2.trials
NUMTRIALS=100
echo " "

for i in `seq 1 $num_trials`
do
  echo "Running trial $i..."
  python wumpus.py NaiveSafe -R 40 -W 2 -P 5 -B 4 -A 3 -M 0.5 -S1 0.85 -S2 0.75 -S3 0.15 -S4 0.10 >> trials/test2.trials
  trap "echo '\nExited test!'; exit;" SIGINT SIGTERM
done

echo "Successes out of $NUMTRIALS:" >> results/test2.results
grep -c "SCREAM" trials/test2.trials >> results/test2.results
echo "\nSuccesses out of $NUMTRIALS:"
echo `grep -c "SCREAM" trials/test2.trials`
echo "\nTest completed!"
