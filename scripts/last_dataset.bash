#!/usr/bin/env bash
# Martin Kersner, m.kersner@gmail.com
# 2016/07/25

# 578fa2f

DATA_PATH="../data/"

OUT=$(mktemp /tmp/output.XXX) || { echo "Failed to create temp file"; exit 1; }
wget "https://numer.ai/" -qO $OUT 
DATASET=`cat $OUT | grep -Eo "leaderboardSince\":\"[0-9]+-[0-9]+-[0-9]+" | grep -Eo "[0-9]+-[0-9]+-[0-9]+"`

DIRECTORY="$DATA_PATH""$DATASET"

if [ ! -d "$DIRECTORY" ]; then
  #mkdir "$DIRECTORY"
  echo "NEW DATASET!"
fi

rm $OUT
