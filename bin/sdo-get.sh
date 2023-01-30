sat=$1;
to="$sat-files.txt";
rm $to;
url="https://sdo.gsfc.nasa.gov/assets/img/browse/$2/$3";
for i in $(mojo get -r -k  "$url" a attr href | egrep -v "([?]|assets)"); do
for f in $(mojo get "$url/$i/" a attr href | egrep 4096_$sat.jpg); do
echo "$url$i$f";
done;
done    



