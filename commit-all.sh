for file in *; do 
    [ -d "$file" ] || continue
    echo "cd $file"; 
    cd ./$file
    git add --all
    git commit -m "feat: update articles"
    git pull
    git push
    cd ..
done
