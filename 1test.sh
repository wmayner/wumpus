#!/bin/bash
# Should run in ~30 seconds for 100 iterations
# TA Success rate: 3763 / 1,000 (~38.0%)
echo " " > trials/test1.trials
NUMTRIALS=1000
echo " "

for i in `seq 1 $NUMTRIALS`
do
  echo "Running trial $i..."
  python wumpus.py NaiveSafe -R 20 -W 1 -A 1 -B 1 -M 0.5 -P 3 -S1 0.9 -S2 0.8 -S3 0.1 -S4 0.05 >> trials/test1.trials
  trap "echo '\nExited test!'; exit;" SIGINT SIGTERM
done

echo "Successes out of $NUMTRIALS:" >> results/test1.results
grep -c "SCREAM" trials/test1.trials >> results/test1.results
echo "\nSuccesses out of $NUMTRIALS:"
echo `grep -c "SCREAM" trials/test1.trials`
echo "\nTest completed!"
