
start=$2
sat=$1
end=$(date +"%Y%m%d")
year=${start:0:4}
while ! [[ $start > $end ]]; do
 
for i in $( mojo get "https://soho.nascom.nasa.gov/data/REPROCESSING/Completed/$year/$sat/$start/" a attr href | egrep "1024");
do
echo "https://soho.nascom.nasa.gov/data/REPROCESSING/Completed/$year/$sat/$start/$i";
done
start=$(date -j -v+1d -f %Y%m%d $start +%Y%m%d)
done
