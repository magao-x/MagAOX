
Copy and paste, then positon the new node.  In the drawio file a basic node will have 5 new components, with numbers for ids in sequential order.

Note: in order to prevent breaking other things (that might have the same numbers in them!) it is recommended that you cut the new components, paste them into a blank document, and make the following changes there.  After you are done, then paste it back.

Note: it is important to use a select-all-and-replace in your text editor to make sure you update all of the places needed.  This will make sure the links, etc., work. However, it is also good practice to inspect each change to make sure only the "id" is being changed (and places where the id is needed, such as when specifying "parent" and "source" and "target"), and not numbers such as heights and positions.

Let's say the node is named "camera". The lowest numbered component will have `style="group"`.  In this example the group mxCell has `id="group:camera"`.  Now perform a select-all-and-replace operation on `"group:camera"`, to replace it with "group:camera".  This will also update the parents of the node's other cells.

Now move the next component, which is the rectangle forming the node.  Here we select-all-and-replace on "node:camera" with "node:camera".

The next cell is the input.  Here we select-all-and-replace "40" with "input:camera:in".  Here you could change the name of the input from "in" if desired (it's usually not).   Be Careful!  This is an example where blindly replacing all "40" in the XML will cause a problem becuase it is also used as a height.

Do the same with "41" to make it "output:camera:out".

Finally we change "42" into "link:camera:in2out".

```
<mxCell id="38" value="" style="group" connectable="0" vertex="1" parent="1">
                    <mxGeometry x="-256" y="392" width="180" height="120" as="geometry"/>
                </mxCell>
                <mxCell id="39" value="camera" style="rounded=0;whiteSpace=wrap;html=1;fontSize=18;" vertex="1" parent="38">
                    <mxGeometry width="180" height="120" as="geometry"/>
                </mxCell>
                <mxCell id="40" value="in" style="rounded=0;whiteSpace=wrap;html=1;strokeColor=#FF0000;" vertex="1" parent="38">
                    <mxGeometry y="80" width="70" height="40" as="geometry"/>
                </mxCell>
                <mxCell id="41" value="out" style="rounded=0;whiteSpace=wrap;html=1;strokeColor=#FF0000;" vertex="1" parent="38">
                    <mxGeometry x="110" y="80" width="70" height="40" as="geometry"/>
                </mxCell>
                <mxCell id="42" value="" style="endArrow=classic;html=1;dashed=1;strokeColor=#00FF00;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;" edge="1" parent="38" source="40" target="41">
                    <mxGeometry width="50" height="50" relative="1" as="geometry"/>
                </mxCell>

```

```
<mxCell id="group:camera" value="" style="group" connectable="0" vertex="1" parent="1">
                    <mxGeometry x="-256" y="392" width="180" height="120" as="geometry"/>
                </mxCell>
                <mxCell id="node:camera" value="camera" style="rounded=0;whiteSpace=wrap;html=1;fontSize=18;" vertex="1" parent="group:camera">
                    <mxGeometry width="180" height="120" as="geometry"/>
                </mxCell>
                <mxCell id="input:camera:in" value="in" style="rounded=0;whiteSpace=wrap;html=1;strokeColor=#FF0000;" vertex="1" parent="group:camera">
                    <mxGeometry y="80" width="70" height="40" as="geometry"/>
                </mxCell>
                <mxCell id="output:camera:out" value="out" style="rounded=0;whiteSpace=wrap;html=1;strokeColor=#FF0000;" vertex="1" parent="group:camera">
                    <mxGeometry x="110" y="80" width="70" height="40" as="geometry"/>
                </mxCell>
                <mxCell id="link:camera:in2out" value="" style="endArrow=classic;html=1;dashed=1;strokeColor=#00FF00;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;" edge="1" parent="group:camera" source="input:camera:in" target="output:camera:out">
                    <mxGeometry width="50" height="50" relative="1" as="geometry"/>
                </mxCell>
```

After renaming everything, connect the input of this node to the output of the previous node with an arrow to create a "beam".  Save the file, and you will have something like:

```
<mxCell id="48" value="" style="endArrow=classic;html=1;strokeColor=#FF0000;strokeWidth=2;fontColor=#FF0000;fontSize=14;entryX=0;entryY=0.5;entryDx=0;entryDy=0;exitX=1;exitY=0.5;exitDx=0;exitDy=0;" edge="1" parent="1" source="output:filter:out" target="input:stageadc1:in">
                    <mxGeometry relative="1" as="geometry" />
                </mxCell>
```

Let's say the previous node is named "filter". Note that the "source" and "target" attributes are filled in.  We just need to edit `id="48"` --> `id="beam:filter2camera"`.
