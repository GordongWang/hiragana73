#!/bin/bash

if [ "$(uname)" == 'Darwin' ]; then
    SED='gsed'
else
    SED='sed'
fi

DATA_SETS=(`find . -name \*.png -print`)

echo "unicode,filename,base64" > ./dataset.csv

COUNT=0

for DATA_SET in ${DATA_SETS[@]}
do
    OUT=$(echo ${DATA_SET} | awk -F '/' '{print $3 "," $4}')
    BASE64=$(base64 --wrap=0 ${DATA_SET})
    echo "${OUT},${BASE64}" >> ./dataset.csv
    COUNT=$(( COUNT + 1 ))
    echo -n "."
done
