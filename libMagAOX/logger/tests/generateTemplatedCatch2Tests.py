#!/bin/env python3

'''
Generate one Catch2 test file per flatlog type from catch2TestTemplate.jinja2.

Input:  every ../types/<log_type>.hpp and its schema ../types/schemas/<log_type>.fbs.
Output: ./generated_tests/<log_type>_generated_tests.cpp, one per type. The folder is
        emptied first and is ignored by git.
Run:    python3 ./generateTemplatedCatch2Tests.py [-s <seed> | -i]   (Python 3.12 or newer)
        or `make generated_tests` in this directory, which uses $(PYTHON) from Make/python.mk. The Makefile here also runs it on
        demand when it builds catch2_tests, which #includes every generated file into one
        binary. tests/testMagAOX.bash runs that binary after the tests in tests/tests.list.

The reference for what gets generated, the options, and the caveats about how field names
in the .hpp and .fbs must line up is flatlogs.md in this directory. The XWCTEST fault
injection macros used by the hand written tests in this directory are a separate
mechanism, documented in tests/genTestMacros.md.
'''

import os
import sys
import subprocess
import glob
import re
import pathlib
import string
import random
import getopt


# Maps each logMeta::valTypes enumerator name to the C++ type that logMeta.cpp casts
# the accessor function pointer of a logMetaDetail to before calling it. The generated
# tests use the same cast, so the two must agree.
VALTYPE_TO_CPP = {
    "String": "std::string",
    "Bool": "bool",
    "Char": "char",
    "UChar": "unsigned char",
    "Short": "short",
    "UShort": "unsigned short",
    "Int": "int",
    "UInt": "unsigned int",
    "Long": "long",
    "ULong": "unsigned long",
    "LongLong": "long long",
    "ULongLong": "unsigned long long",
    "Float": "float",
    "Double": "double",
    "Vector_String": "std::vector<std::string>",
    "Vector_Bool": "std::vector<bool>",
    "Vector_Char": "std::vector<char>",
    "Vector_UChar": "std::vector<unsigned char>",
    "Vector_Short": "std::vector<short>",
    "Vector_UShort": "std::vector<unsigned short>",
    "Vector_Int": "std::vector<int>",
    "Vector_UInt": "std::vector<unsigned int>",
    "Vector_Long": "std::vector<long>",
    "Vector_ULong": "std::vector<unsigned long>",
    "Vector_LongLong": "std::vector<long long>",
    "Vector_ULongLong": "std::vector<unsigned long long>",
    "Vector_Float": "std::vector<float>",
    "Vector_Double": "std::vector<double>",
}

gNextVals = {
    "string" : 0,
    "int64"  : 0,
    "uint64" : 0,
    "int32"  : 0,
    "uint32" : 0,
    "int16"  : 0,
    "uint16" : 0,
    "int8"   : 0,
    "uint8"  : 0,
    "float"  : 0,
    "double" : 0
}
gIncrementingVals = False

# check jinja2 is installed. install it if not
try:
    import jinja2
except ModuleNotFoundError:
    print("module 'Jinja2' is not installed. Installing Jinja2...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'Jinja2'])
    import jinja2


'''
Get base type of log. This is needed for log types that inherit from a base type
that specfies the messageT(...)
'''
def getBaseType(lines : list) -> str:
    # use regex to find #include "<baseType>.hpp"
    baseType = ""
    for line in lines:
        match = re.search(r'^struct [a-z_]* : public [a-z_]*', line)
        if match != None:
            baseType = line.strip().split()[-1]
            baseType = baseType.split("<")[0]

    return baseType


'''
NOTE: This relies on name order in .fbs schema and .hpp files to be the same.
'''
def getSchemaFieldInfo(fname : str) -> tuple[str, tuple] :
    schemaFolderPath = "./../types/schemas/"
    schemaFolderPath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), schemaFolderPath)
    )

    schemaFilePath = os.path.join(schemaFolderPath, f"{fname}.fbs")
    if not os.path.isfile(schemaFilePath):
        return "", tuple()

    schemaFile = open(schemaFilePath, "r")

    schemaFieldInfo = []
    subTables = dict() # dict where key is sub-table name, value is [(fieldname, type)...]
    curSubTable = None
    inTable = False
    schemaTableName = ""
    for line in schemaFile:
        if "table" in line:
            # check if `table <log_type>_fb`
            match = re.search(r'^table [a-zA-Z_]*_fb', line)
            if match != None:
                line = line.strip().split()
                tableIdx = line.index("table")
                schemaTableName =  line[tableIdx + 1]
            # otherwise it is a sub-table
            else:
                line = line.strip().split()
                subNameIdx = line.index("table")
                subName =  line[subNameIdx + 1]
                subTables[subName] = []
                curSubTable = subName # we are in a sub-table of the schema

        if not inTable and "{" in line:
            inTable = True
            continue

        if inTable:
            line = line.strip()
            if ("//" in line):
                continue

            if ("}" in line):
                inTable = False
                curSubTable = None
                continue

            if ("deprecated" in line):
                continue

            if (line != ""):
                fieldParts = line.strip().rstrip(";").split(":")
                name = fieldParts[0]
                fieldType = fieldParts[1].split()[0]

                if curSubTable is not None:
                    # add to subtable dict for now, will be added in later
                    subTables[curSubTable].append((name, fieldType))
                else:
                    schemaFieldInfo.append((name, fieldType))
                continue

    if len(subTables) == 0:
        return schemaTableName, tuple(schemaFieldInfo)


    # go through sub tables and add them in
    newSchemaFieldInfo = []
    for field in schemaFieldInfo:
        fieldType = field[1]
        if fieldType in subTables.keys():
               newSchemaFieldInfo.append({field[0] : subTables[fieldType]})
        else:
            newSchemaFieldInfo.append(field)
    # print(newSchemaFieldInfo)
    return schemaTableName, tuple(newSchemaFieldInfo)


'''
Quick check that the types in .fbs correspond, mainly strings match to strings,
and vectors to vectors.
If they do not correspond, the behavior for comparing the fb values in the tests
is undefined, and action beyond this generator will need to be taken.
'''
def typesCorrespond(fbsType : str, cType : str) -> bool:
    if ("[" in fbsType) or ("vector" in cType):
        return ("[" in fbsType) and ("vector" in cType)

    if ("string" in fbsType) or ("string" in cType or "char *" in cType):
        return (("string" in fbsType) and ("string" in cType or "char *" in cType))

    return True


'''
Check it is not a base log type.
Must have eventCode and defaultLevel
'''
def isValidLogType(lines : list) -> bool:
    hasEventCode = False
    hasDefaultLevel = False
    for line in lines:

        # check event code
        eventCode = re.search("flatlogs::eventCodeT eventCode = eventCodes::[A-Za-z_0-9]*;", line)
        if eventCode != None:
            hasEventCode = True

        # check default level
        defaultLevel = re.search("flatlogs::logPrioT defaultLevel = flatlogs::logPrio::[A-Za-z_0-9]*;", line)
        if defaultLevel != None:
            hasDefaultLevel = True

        # if we have both already, return
        if hasEventCode and hasDefaultLevel:
            return True

    return (hasEventCode and hasDefaultLevel)

'''
Parse the getAccessor(const std::string &member) implementation of a log type header.
For each recognized member it recovers the member name, the C++ return type, and the
accessor function name that getAccessor() registers. getAccessor() takes the address of
every per-field accessor function regardless of which branch runs at runtime. So the type
header itself is the source of truth for which accessors exist and how logMeta.cpp would
cast and invoke each one. This avoids deriving a second type mapping from the field types
directly, which could disagree with the first.
'''
def getAccessorFieldInfo(headerText : str) -> list:
    results = []
    branchPattern = re.compile(
        r'member\s*==\s*"([A-Za-z0-9_]+)"\s*\)\s*\{?\s*return\s+logMetaDetail\(\s*\{(.*?)\}\s*\)\s*;',
        re.DOTALL
    )
    for branchMatch in branchPattern.finditer(headerText):
        member = branchMatch.group(1)
        body = branchMatch.group(2)

        valTypeMatch = re.search(r'logMeta::valTypes::(\w+)', body)
        # The '&' before the function name is usually present. A bare function name
        # already decays to its address, so it is sometimes omitted, for example in the
        # "water" accessor of ocam_temps. Both forms are tolerated.
        funcMatch = re.search(r'reinterpret_cast\s*<\s*void\s*\*\s*>\s*\(\s*&?\s*(\w+)\s*\)', body)
        if valTypeMatch is None or funcMatch is None:
            continue

        cppType = VALTYPE_TO_CPP.get(valTypeMatch.group(1))
        if cppType is None:
            continue

        results.append({"member": member, "cppType": cppType, "funcName": funcMatch.group(1)})

    return results

# Maps each fixed-width integer typedef to the plain C++ type it resolves to on this
# platform. This lets a vector element type taken from a field declaration compare equal
# to the element type in VALTYPE_TO_CPP.
TYPEDEF_ALIASES = {
    "uint8_t": "unsigned char",
    "int8_t": "char",
    "uint16_t": "unsigned short",
    "int16_t": "short",
    "uint32_t": "unsigned int",
    "int32_t": "int",
    "uint64_t": "unsigned long",
    "int64_t": "long",
}

# Strips whitespace and replaces a fixed-width typedef with its plain type.
def normalizeCppType(t : str) -> str:
    t = t.strip()
    return TYPEDEF_ALIASES.get(t, t)

'''
Some accessor functions pass through the raw flatbuffer value of a field. Those are safe
to compare against the same raw value collected in the constructor. Others compute a
derived or formatted value with a different C++ type. For example an integer state field
may have a string formatting accessor, and a vector<uint8_t> field may have a vector<bool>
accessor. Comparing those against the raw field value would not compile or would not be
meaningful. This function decides, for one accessor and field pairing, whether an exact
value REQUIRE is safe to emit. When it is not safe, the generated test still invokes the
accessor to cover its body, without asserting a specific return value.
'''
def valuesComparable(accessorCppType : str, field : dict) -> bool:
    fieldType = field["type"]

    if accessorCppType == "std::string":
        return "string" in fieldType

    if accessorCppType.startswith("std::vector<"):
        if "std::vector" not in fieldType:
            return False
        elemType = normalizeCppType(accessorCppType[len("std::vector<"):-1])
        vecType = normalizeCppType(field.get("vectorType", ""))
        return elemType == vecType

    # A Bool-valued accessor wrapping a wider integral flag field is not necessarily a
    # plain truthy pass-through. Some types instead check whether the field equals 1, as
    # a tri-state convention where 0 and 1 are values and anything else means unset. Both
    # patterns exist in this codebase. They cannot be told apart without executing the
    # accessor. So equality is only asserted here when the field is itself already a real
    # bool, which is an exact and unambiguous match. Otherwise the accessor is still
    # invoked for coverage without asserting its value.
    if accessorCppType == "bool":
        return fieldType == "bool"

    # This is a plain scalar such as char, int, or float. Implicit conversions make these
    # safe to compare against any other scalar field, but not against string or vector
    # fields.
    if "string" in fieldType or "vector" in fieldType:
        return False
    return True

'''
For each generated messageT variant, which is each entry in messageTypes, build the list
of accessor checks to emit. Every accessor that getAccessor() registers gets invoked, so
its body is exercised for coverage regardless of naming. The exact value REQUIRE is only
emitted when the accessor function name matches one of the constructor field names of
this variant and the types are comparable. For example the "tempStatus" accessor of
telem_stdcam has no "tempStatus" constructor field. The constructor names it "status"
and routes it through a differently named meta keyword, so there is no raw value here to
compare against. The "pokes" constructor variant of telem_pokecenter never sets poke_x or
poke_y as named constructor fields either. In both cases the accessor is still called.
'''
def filterAccessorChecks(accessorFieldInfo : list, messageTypes : list) -> list:
    result = []
    for msgType in messageTypes:
        fieldByName = { f["name"]: f for f in msgType }
        seen = set()
        checks = []
        for accessor in accessorFieldInfo:
            if accessor["funcName"] in seen:
                continue
            seen.add(accessor["funcName"])

            accessor = dict(accessor)
            field = fieldByName.get(accessor["funcName"])
            accessor["compareValue"] = ( field is not None
                                          and "char *" not in field["type"]
                                          and valuesComparable(accessor["cppType"], field) )
            checks.append(accessor)
        result.append(checks)
    return result

def makeTestInfoDict(hppFname : str, baseTypesDict : dict) -> dict:
    returnInfo = dict()
    headerFile = open(hppFname,"r")
    headerLines = headerFile.readlines()

    # add name of test/file/type to be generated
    fNameParts = hppFname.split("/")
    returnInfo["name"] = fNameParts[-1].strip().split(".")[0]
    CamelCase = "".join([word.capitalize() for word in returnInfo["name"].split("_")])
    returnInfo["nameCamelCase"] = CamelCase[0].lower() + CamelCase[1:]
    # print(f"LOGNAME: {returnInfo["name"]}")
    returnInfo["genTestFname"] = f"{returnInfo['name']}_generated_tests.cpp"
    returnInfo["className"] = "C" + "".join([word.capitalize() for word in returnInfo["name"].split("_")])
    returnInfo["classVarName"] = "".join([word[0].lower() for word in returnInfo["name"].split("_")])
    returnInfo["baseType"] = getBaseType(headerLines)
    returnInfo["hasGeneratedHfile"] = hasGeneratedHFile(returnInfo["name"])

    # cannot generate tests from this file alone, need base type
    if not isValidLogType(headerLines):
        if returnInfo["name"] not in baseTypesDict:
            baseTypesDict[returnInfo["name"]] = set()

        return None # don't render anything from this file

    # iterate through all lines in header to:
    # 1. find where messageT structs are being made -> describes fields
    # 2. check that is has its own <Get|Create|Verify><name>_fb methods
    fbMethodName = f"Create{returnInfo["name"][0].upper() + returnInfo["name"][1:]}_fb"
    hasFbMethods = False
    messageStructIdxs = []
    for i in range(len(headerLines)):
        if "messageT(" in headerLines[i]:
            messageStructIdxs.append(i)
        if fbMethodName in headerLines[i]:
            hasFbMethods = True

    schemaTableName, schemaFieldInfo = getSchemaFieldInfo(returnInfo["name"])
    returnInfo["schemaTableName"] = schemaTableName

    # handle log types that inherit from base types
    if len(messageStructIdxs) == 0:

        if returnInfo["baseType"] not in baseTypesDict:
            baseTypesDict[returnInfo["baseType"]] = set()

        # add inhertied type to dict where val is the base type it inherits from
        baseTypesDict[returnInfo["baseType"]].add(returnInfo["name"])

        return None # don't render me yet!

    # if it does not have its own fb method, find name of class its using
    if not hasFbMethods:
        for line in headerLines:
            if re.search("^.*Create[a-zA-Z_]*_fb.*$", line) and returnInfo["schemaTableName"] == "":
                # figure out name of fb methods this type is re-using, e.g. ao_observer -> observer
                startIndex = line.find("Create") + len("Create")
                endIndex = line.find("_fb")
                returnInfo["schemaTableName"] = f"{line[startIndex:endIndex]}_fb"

    returnInfo["messageTypes"] = getMessageFieldInfo(messageStructIdxs, headerLines, schemaFieldInfo)

    # Collect the accessors this type registers, then decide per messageT variant which
    # of them can be checked against a raw field value in the generated test.
    accessorFieldInfo = getAccessorFieldInfo("".join(headerLines))
    returnInfo["accessorChecksPerMsgType"] = filterAccessorChecks(accessorFieldInfo, returnInfo["messageTypes"])

    return returnInfo

'''
Parse out field type and name from string
'''
def getTypeAndName(fieldParts : list) -> tuple[str, str]:

    typeIdxStart = 1 if (fieldParts[0] == "const") else 0
    fieldType = fieldParts[typeIdxStart]

    if fieldParts[typeIdxStart + 1] == "&":
        nameIdx = (typeIdxStart + 2)
    elif fieldParts[typeIdxStart + 1] == "*":
        nameIdx = (typeIdxStart + 2)
        fieldType += " *"
    else:
        nameIdx = (typeIdxStart + 1)

    name = fieldParts[nameIdx].rstrip(")").rstrip(",")

    if name[0] == "*":
        fieldType += " *"

    name = name.lstrip("&*")

    return fieldType, name

'''
Checks if log type has a corresponding generated .h file in ./types/generated
'''
def hasGeneratedHFile(logName : str) -> bool:
    generatedFolderPath = "./../types/generated/"
    generatedFolderPath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), generatedFolderPath)
    )

    generatedFilePath = os.path.join(generatedFolderPath, f"{logName}_generated.h")
    if os.path.isfile(generatedFilePath):
        return True

    return False

def getIntSize(type : str) -> int:
    intSizeBits = 32 # default size 32 bits
    if "_t" in type:
        typeParts = type.split("_t")
        intSizeBits = int(typeParts[0][-1]) if (int(typeParts[0][-1]) == 8) \
                    else int(typeParts[0][-2:])

    return intSizeBits


def getRandInt(type : str) -> int:
    unsigned = True if "uint" in type else False

    intSizeBits = getIntSize(type)

    if not unsigned:
        intSizeBits -= 1

    max = (2 ** intSizeBits) - 1
    min = 0 if unsigned else (0 - max - 1)

    return random.randint(min, max)

def getIncrementingInt(type : str) -> int:
    intSizeBits = getIntSize(type)

    max = (2 ** intSizeBits) - 1

    if  "int8_t" in type:
        gNextVals["int8"] =  (gNextVals["int8"]     + 1) % max
        return gNextVals["int8"]
    elif   "uint8_t" in type:
        gNextVals["uint8"] =  (gNextVals["uint8"]   + 1) % max
        return gNextVals["uint8"]
    elif  "int16_t" in type:
        gNextVals["int16"] =  (gNextVals["int16"]   + 1) % max
        return gNextVals["int16"]
    elif "uint16_t" in type:
        gNextVals["uint16"] = (gNextVals["uint16"]  + 1) % max
        return gNextVals["uint16"]
    elif  "int32_t" in type:
        gNextVals["int32"] =  (gNextVals["int32"]   + 1) % max
        return gNextVals["int32"]
    elif "uint32_t" in type:
        gNextVals["uint32"] = (gNextVals["uint32"] + 1) % max
        return gNextVals["uint32"]
    elif  "int64_t" in type:
        gNextVals["int64"] =  (gNextVals["int64"]  + 1) % max
        return gNextVals["int64"]
    elif "uint64_t" in type:
        gNextVals["uint64"] = (gNextVals["uint64"] + 1) % max
        return gNextVals["uint64"]
    else:
        gNextVals["int32"] =  (gNextVals["int32"]  + 1) % max
        return gNextVals["int32"]

def getTestValFromType(fieldType : str, schemaFieldType = None) -> str:
    if "bool" in fieldType or (schemaFieldType is not None and "bool" in schemaFieldType):
        return "1"
    elif "string" in fieldType or "char *" in fieldType:
        if gIncrementingVals:
            gNextVals["string"] += 1
            return f'"{gNextVals["string"]}"'
        randString = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
        return f'"{randString}"'
    elif "int" in fieldType:
        if gIncrementingVals:
            return str(getIncrementingInt(fieldType))
        # need 'u' suffix for randomly generated uint64_t to avoid:
        # "warning: integer constant is so large that it is unsigned"
        return f'{str(getRandInt(fieldType))}u' if "uint64_t" in fieldType else str(getRandInt(fieldType))
    elif "float" in fieldType:
        if gIncrementingVals:
            gNextVals["float"] += 1
            return str(round( (gNextVals["float"] / 100000), 6))
        return str(round(random.random(), 6))
    elif "double" in fieldType:
        if gIncrementingVals:
            gNextVals["double"] += 1
            return str(round( (gNextVals["double"] / 10000000000), 14))
        return str(round(random.random(), 14))
    else:
        return "{}"


def makeTestVal(fieldDict : dict) -> str:
    if "vector" in fieldDict["type"]:
        vals = [ getTestValFromType(fieldDict["vectorType"]) for i in range(10)]

        # special case telem_pokecenter because vector follows specific format
        if fieldDict["name"] == "pokes" and "vector<float" in fieldDict["type"]:
            catchAssertVals = [vals[i] for i in range(0, len(vals), 2)]
            fieldDict["specialAssertVal"] = f"{{ {",".join(catchAssertVals)} }}"
        return f"{{ {",".join(vals)} }}"

    if "schemaType" in fieldDict:
        return getTestValFromType(fieldDict["type"], fieldDict["schemaType"])

    return getTestValFromType(fieldDict["type"])

# returns tuple of schema field info, subtable name (none if no subtable)
def findMatchingSchemaField(schemaFieldInfo, fieldName):
    for schemaField in schemaFieldInfo:
        if isinstance(schemaField, tuple) and schemaField[0] == fieldName:
                return schemaField, None
        if isinstance(schemaField, dict):
            subTableName = next(iter(schemaField))
            for subField in schemaField[subTableName]:
                if len(subField) != 2:
                    continue
                if subField[0] == fieldName:
                    return subField, subTableName
    return None, None # no matching field in schema for given fieldName

def setDefaultArgOfLastField(fieldsList, fieldParts):
    # this is the default arg value for msgsFieldList[-1]
    fieldsList[-1]["defaultArg"] = " ".join(fieldParts).strip("=").strip()

    # std::source_location aliases separate fields file and line for software_log
    if "std::source_location" in fieldsList[-1]["type"]:
        if fieldsList[-1]["name"] == "loc":  # can remove this line if we don't want "loc" strictly associated with alias
            fieldsList.pop()

            # special case software log loc alias for file and line fields.
            fieldsList.append(
                {
                    "type": "char *",
                    "name": "file",
                    "schemaName": "file",
                    "schemaType": "string",
                    "testVal": None,
                    "defaultArg": True,
                    "defaultTestVal": "__FILE__"
                }
            )
            fieldsList.append(
                {
                    "type": "uint32_t",
                    "name": "line",
                    "schemaName": "line",
                    "schemaType": "uint32",
                    "testVal": None,
                    "defaultArg": True
                }
            )



'''
make 2d array. each inner array contains dictionaries corresponding to
the type(s) and name(s) of field(s) in a message:
[ [ {type : x, name: y ...}, {name: type, ...} ], ... ]
'''
def getMessageFieldInfo(messageStructIdxs: list, lines : list, schemaFieldInfo : tuple):
    msgTypesList = []
    subTableDictIndex = 0

    # extract log field types and names
    inMultilineComment = False
    inDefaultArgDef = False
    for i in range(len(messageStructIdxs)):
        structIdx = messageStructIdxs[i]
        msgsFieldsList = []

        closed = False
        fieldCount = 0
        while not closed and structIdx < len(lines):

            line = lines[structIdx]

            # check if this is a closing line
            if ")" in line:
                if ("//" in line and line.find(")") > line.find("//")):
                    # parenthesis is in comment
                    pass
                elif line.strip().strip(")") == "":
                    break # field is done
                else:
                    openParenCount = line.count("(")
                    closeParenCount = line.count(")")
                    # check if truly closed or not
                    if (closeParenCount > openParenCount) or \
                       (closeParenCount == openParenCount and "messageT(" in line):
                        closed = True # parse the field, don't leave loop yet
                        line = line[:line.rfind(")")]

            if inMultilineComment:
                if "*/" in line:
                    inMultilineComment = False
                    structIdx += 1
                continue


            # trim line to just get field info
            indexStart = (line.find("messageT(") + len("messageT(")) if "messageT(" in line else 0

            indexEnd = len(line)
            # adjust for comments. Note /* takes precedence over //
            if "/*" in line and line.find("/*") < indexEnd:
                indexEnd = line.find("/*")
                # handle multiline comments
                if "*/" not in line:
                    inMultilineComment = True
            elif "//" in line:
                indexEnd = line.find("//")


            line = line[indexStart:indexEnd].strip()

            fieldParts =  [part.strip().split() for part in line.strip().rstrip(",").split(",")]

            for field in fieldParts:
                fieldDict = {}

                if len(field) > 0 and "//" in field[0]:
                    break
                if len(field) == 0:
                    break

                # check if this is a default arg value
                if inDefaultArgDef and len(msgsFieldsList) > 1:
                    setDefaultArgOfLastField(msgsFieldsList, field)
                    inDefaultArgDef = False
                    field.pop(0)
                    if len(field) == 0:
                        break

                # handle default arguments that expand across two lines
                if field[0] == "=":
                    setDefaultArgOfLastField(msgsFieldsList, field)
                    continue
                if field[-1] == '=':
                    # set flag but still need to parse this fields info.
                    # don't leave this loop iteration yet.
                    inDefaultArgDef = True


                # find type and name
                fieldType, name = getTypeAndName(field)

                fieldDict["type"] = fieldType
                fieldDict["name"] = name
                # get vector type if necessary
                if "std::vector" in fieldDict["type"]:
                    typeParts = fieldDict["type"].split("<")
                    vectorIdx = [i for i, e in enumerate(typeParts) if "std::vector" in e][0]
                    vectorType = typeParts[vectorIdx + 1].strip(">")
                    fieldDict["vectorType"] = vectorType

                if len(schemaFieldInfo) != 0:

                    if isinstance(schemaFieldInfo[fieldCount], tuple):
                        fieldDict["schemaName"] = schemaFieldInfo[fieldCount][0]
                        fieldDict["schemaType"] = schemaFieldInfo[fieldCount][1]
                        fieldCount += 1

                        # check if matching name in schema file exists, let this overwrite
                        matchingSchemaField, subTableName = findMatchingSchemaField(schemaFieldInfo, fieldDict["name"])
                        if matchingSchemaField != None and len(matchingSchemaField) == 2:
                            subTableStr = f"{subTableName}()->" if subTableName is not None else ""
                            fieldDict["schemaName"] = f"{subTableStr}{matchingSchemaField[0]}"
                            fieldDict["schemaType"] = matchingSchemaField[1]

                    else:
                        # go into dictionary..
                        subTableName = next(iter(schemaFieldInfo[fieldCount]))
                        schemaFieldName = schemaFieldInfo[fieldCount][subTableName][subTableDictIndex][0]
                        schemaFieldType = schemaFieldInfo[fieldCount][subTableName][subTableDictIndex][1]
                        fieldDict["schemaName"] = f"{subTableName}()->{schemaFieldName}"
                        fieldDict["schemaType"] = schemaFieldType
                        subTableDictIndex += 1
                        if (subTableDictIndex >= len(schemaFieldInfo[fieldCount][subTableName])):
                            # reset dictionary index if we need to
                            subTableDictIndex = 0
                            fieldCount += 1

                    # check schemaType correlates to type in .hpp file
                    if not typesCorrespond(fieldDict["schemaType"], fieldDict["type"]):
                        # if types don't correspond, then use name in messageT and hope for best.
                        # this is why if types are different, then names MUST correspond between
                        # .fbs and .hpp file
                        del fieldDict["schemaName"]
                        del fieldDict["schemaType"]

                fieldDict["testVal"] = makeTestVal(fieldDict)

                # add field dict to list of fields
                msgsFieldsList.append(fieldDict)

                # note we do this after the fieldDict has been appended to msgsFieldList.
                # This is because the function setDefaultArgOfLastField() does some
                # special casing to replace std::current_location with field and line
                # within the msgsFieldsList
                if "=" in field:
                    setDefaultArgOfLastField(msgsFieldsList, field)

            structIdx += 1

        msgTypesList.append(msgsFieldsList)

    return msgTypesList

def makeInheritedTypeInfoDict(typesFolderPath : str, baseName : str, logName : str) -> dict:
    returnInfo = dict()

    baseFilePath = os.path.join(typesFolderPath, f"{baseName}.hpp")
    baseHFile = open(baseFilePath,"r")

    # add name of test/file/type to be generated
    # print(f"LOGNAME: {logName}")
    returnInfo["name"] = logName
    returnInfo["genTestFname"] = f"{returnInfo['name']}_generated_tests.cpp"
    returnInfo["className"] = "C" + "".join([word.capitalize() for word in returnInfo["name"].split("_")])
    CamelCase = "".join([word.capitalize() for word in returnInfo["name"].split("_")])
    returnInfo["nameCamelCase"] = CamelCase[0].lower() + CamelCase[1:]
    returnInfo["classVarName"] = "".join([word[0].lower() for word in returnInfo["name"].split("_")])
    returnInfo["baseType"] = baseName
    returnInfo["hasGeneratedHfile"] = hasGeneratedHFile(logName)

    baseHLines = baseHFile.readlines()

    # find where messageT structs are being made in base log file -> describes fields
    messageStructIdxs = []
    for i in range(len(baseHLines)):
        if "messageT(" in baseHLines[i]:
            messageStructIdxs.append(i)

    schemaTableName, schemaFieldInfo = getSchemaFieldInfo(baseName)

    returnInfo["schemaTableName"] = schemaTableName
    msgFieldInfo = getMessageFieldInfo(messageStructIdxs, baseHLines, schemaFieldInfo)

    returnInfo["messageTypes"] = [[]] if "empty_log" in baseName else msgFieldInfo

    # getAccessor() is not picked up by the base messageT parsing above. If the own header
    # of logName overrides getAccessor(), that is what MagAOX::logger::<logName>::getAccessor
    # actually resolves to. Otherwise it resolves to the version in the base type.
    ownFilePath = os.path.join(typesFolderPath, f"{logName}.hpp")
    accessorFieldInfo = []
    if os.path.isfile(ownFilePath):
        with open(ownFilePath, "r") as ownFile:
            accessorFieldInfo = getAccessorFieldInfo(ownFile.read())
    if len(accessorFieldInfo) == 0:
        accessorFieldInfo = getAccessorFieldInfo("".join(baseHLines))
    returnInfo["accessorChecksPerMsgType"] = filterAccessorChecks(accessorFieldInfo, returnInfo["messageTypes"])

    return returnInfo

def versionAsNumber(major, minor):
    return (major * 1000 + minor)

def main():
    # check python version >= 3.9
    if (versionAsNumber(sys.version_info[0], sys.version_info[1]) < versionAsNumber(3,9)):
        print("Error: Python version must be >= 3.9")
        exit(0)


    global gIncrementingVals
    gIncrementingVals = False

    # getopt for random seed or incrementing vals
    try:
        opts, args = getopt.getopt(sys.argv[1:], "is:")
        if len(opts) > 1:
            print("Error: Only one option allowed. -s <seed> or -i for incrementing values.")
            exit(0)

    except getopt.GetoptError:
        print("Usage: python3 ./generateTemplatedCatch2Tests.py -s <seed> | -i")
        exit(0)
    for opt, arg in opts:
        if opt in ["-s"]:
            if not arg.isdigit():
                print(f"Error: random seed {arg} provided is not an integer.")
                exit(0)
            # use random seed if provided with -s
            random.seed(int(arg))
        if opt in ["-i"]:
            gIncrementingVals = True

    # load template
    env = jinja2.Environment(
        loader = jinja2.FileSystemLoader(searchpath=os.path.dirname(__file__))
    )
    env.trim_blocks = True
    env.lstrip_blocks = True

    catchTemplate = env.get_template("catch2TestTemplate.jinja2")

    # path to .hpp files here
    typesFolderPath = "./../types"
    typesFolderPath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), typesFolderPath)
    )

    # generated tests output path
    generatedTestsFolderPath = "./generated_tests/"
    generatedTestsFolderPath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), generatedTestsFolderPath)
    )

    # make directory if it doesn't exist
    pathlib.Path(generatedTestsFolderPath).mkdir(exist_ok=True)
    oldFiles = glob.glob(os.path.join(generatedTestsFolderPath, "*"))
    for file in oldFiles:
        os.remove(file)

    types = os.listdir(typesFolderPath)
    types.sort()
    baseTypesDict = dict() # map baseTypes to the types that inherit from them
    print("generating tests for...")
    for type in types:

        print(type)
        # check valid type to generate tests for
        if ".hpp" not in type:
            continue

        typePath = os.path.join(typesFolderPath, type)

        # make dictionary with info for template
        info = makeTestInfoDict(typePath, baseTypesDict)
        if (info is None):
            # empty dictionary, no tests to make
            continue

        # render
        renderedHeader = catchTemplate.render(info)

        # write generated file
        outPath = os.path.join(generatedTestsFolderPath, info["genTestFname"])
        with open(outPath,"w") as outfile:
            print(renderedHeader,file=outfile)

    # handle types that inherit from baseTypes
    for baseType, inheritedTypes in baseTypesDict.items():

        if len(inheritedTypes) == 0:
            continue

        for inheritedType in inheritedTypes:
            info = makeInheritedTypeInfoDict(typesFolderPath, baseType, inheritedType)
            if (info is None):
                # empty dictionary, no tests to make
                continue

            # render
            renderedHeader = catchTemplate.render(info)

            # write generated file
            outPath = os.path.join(generatedTestsFolderPath, info["genTestFname"])
            with open(outPath,"w") as outfile:
                print(renderedHeader,file=outfile)


if (__name__ == "__main__"):
    main()
