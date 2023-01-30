

for sat in 1600 1700; do
for y in {2020..2022}; do
for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
outfile="../lists/$sat-$y-$m.txt"


if [ ! -f "$outfile" ]; then
  echo " bash sdo-get.sh $sat $y $m" | sh -x > $outfile;
fi
done;
done;
done;
