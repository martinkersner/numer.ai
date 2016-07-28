#!/usr/bin/env bash
# Martin Kersner, m.kersner@gmail.com
# 2016/07/25

# ID
# 2016-07-21 578fa2f
# 2016-07-28 57991fc

# https://datasets.numer.ai/<ID>/numerai_datasets.zip

DATA_PATH="../data/"

OUT=$(mktemp /tmp/output.XXX) || { echo "Failed to create temp file"; exit 1; }
wget "https://numer.ai/" -qO $OUT 
DATASET=`cat $OUT | grep -Eo "leaderboardSince\":\"[0-9]+-[0-9]+-[0-9]+" | grep -Eo "[0-9]+-[0-9]+-[0-9]+"`

DIRECTORY="$DATA_PATH""$DATASET"

if [ ! -d "$DIRECTORY" ]; then
  #mkdir "$DIRECTORY"
  echo "NEW DATASET! $DATASET"
fi

rm $OUT
