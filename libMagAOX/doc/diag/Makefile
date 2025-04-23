########################################################################
# export diagrams to png with drawio CLI                               #
########################################################################

all: png/xwcapp.png png/xwcapp_fsm.png png/xwcapps_connections.png

png/xwcapp.png: xwcapp.drawio
	drawio -t -f png --export -o png/ xwcapp.drawio

png/xwcapp_fsm.png: xwcapp_fsm.drawio
	drawio -t -f png --export -o png/ xwcapp_fsm.drawio

png/xwcapps_connections.png: xwcapps_connections.drawio
	drawio -t -f png --export -o png/ xwcapps_connections.drawio

#note: no clean.  we commit the pngs so dox can be built without running drawio
