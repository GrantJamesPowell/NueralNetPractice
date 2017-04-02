wget https://www.ssa.gov/oact/babynames/state/namesbystate.zip

unzip namesbystate.zip

for f in *.TXT; do 
    mv -- "$f" "${f%.TXT}.data"
done

rm namesbystate.zip
rm StateReadMe.pdf

