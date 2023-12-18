
# Using valgrind for testing

Turn on debug in local/common.mk by adding the line
OPTIMIZE=-g 

do 

```
make all_clean
```

followed by
```
sudo make install
```

Once installed, you have to change the setuid bit for the executable you want to test, e.g.

```
sudo chmod -s /opt/MagAOX/bin/magAOXMaths
```

You will need to be root to run without setuid.  
Then you can run that one executable with

```
valgrind --leak-check=yes --show-leak-kinds=all --trace-children=yes /opt/MagAOX/bin/magAOXMaths -n maths_y

```
