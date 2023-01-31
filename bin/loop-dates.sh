

for sat in 1600; do
for y in {2011..2022}; do
for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
outfile="../lists/$sat-$y-$m.txt"
minimumsize=100
actualsize=$(du -k "$outfile" | cut -f 1)

if [ ! -f "$outfile" ]; then
  echo " bash sdo-get.sh $sat $y $m" | sh -x > $outfile;
else
  if [ $actualsize -ge $minimumsize ]; then
      echo size is over $minimumsize kilobytes
  else
      echo " bash sdo-get.sh $sat $y $m" | sh -x > $outfile;
  fi
fi
done;
done;
done;
