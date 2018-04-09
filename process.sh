while [ true ]
do
    # echo "* Main loop..."
    TODO_FILES=~/entrack/configs/todo/*.yaml
    DOING_FOLDER="configs/doing"
    DONE_FOLDER="configs/done"

    for f in $TODO_FILES
    do
        name=`basename $f`
        if [ -f $f ]; then
            echo "[*] $name ($f)"
            mv $f $DOING_FOLDER/ && \
            smt run --config $DOING_FOLDER/$name -S /local/dhaziza/data -a fit && \
            mv $DOING_FOLDER/$name $DONE_FOLDER/
            sleep 5
        fi
    done
    sleep 30
done
