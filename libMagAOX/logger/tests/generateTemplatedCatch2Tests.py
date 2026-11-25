#!/bin/env python3

'''Generate Catch2 serialization tests; see flatlogs.md for usage.

Constructor parameters follow schema order unless an exact field name matches.
Struct vectors and timestamps retain their component metadata for comparisons.
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

# Inputs with semantic constraints that cannot be inferred from scalar types.
gTestFixtures = {
    "telem_pupilfit": {
        "numPupils": "3",
        "valid": "true",
        "pupils": "{{3, 6, 4, 100, 10}, {6, 12, 8, 110, 10}, {9, 18, 12, 120, 10}}"
    },
    "telem_pupilfit_state": {
        "setpoints": "{{3, 6, 4}, {6, 12, 8}, {9, 18, 12}}",
        "requestedMode": "2",
        "activeMode": "2"
    }
}

# Check Jinja2 is installed; retain the existing installation fallback.
try:
    import jinja2
except ModuleNotFoundError:
    print("module 'Jinja2' is not installed. Installing Jinja2...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'Jinja2'])
    import jinja2

def getBaseType(lines : list) -> str:
    """Find the base log type supplying inherited message constructors."""
    baseType = ""
    for line in lines:
        match = re.search(r'^struct [a-z_]* : public [a-z_]*', line)
        if match != None:
            baseType = line.strip().split()[-1]
            baseType = baseType.split("<")[0]

    return baseType


def getSchemaFieldInfo(fname : str) -> tuple[str, tuple, dict]:
    """Read root fields, nested tables, and qualified struct-vector metadata."""
    schemaFilePath = pathlib.Path(__file__).parent / "../types/schemas" / f"{fname}.fbs"
    if not schemaFilePath.is_file():
        return "", tuple(), {}

    schema = schemaFilePath.read_text()
    schema = re.sub(r"/\*.*?\*/|//[^\n]*", "", schema, flags=re.S)
    root = re.search(r"\broot_type\s+(\w+)\s*;", schema)
    schemaTableName = root[1] if root else ""
    namespace = re.search(r"\bnamespace\s+([\w.]+)\s*;", schema)
    prefix = namespace[1].replace(".", "::") + "::" if namespace else ""
    tables = {}
    structs = {}
    for kind, name, body in re.findall(r"\b(table|struct)\s+(\w+)\s*\{([^}]*)\}", schema):
        fields = []
        for declaration in body.split(";"):
            if "deprecated" in declaration:
                continue
            field = re.match(r"\s*(\w+)\s*:\s*([\w.]+|\[[\w.]+\])", declaration)
            if field:
                fields.append((field[1], field[2]))
        if kind == "struct":
            structs[name] = {"type": prefix + name, "members": fields}
        else:
            tables[name] = fields

    # Constructor parameters stored in nested tables retain their existing order.
    schemaFieldInfo = []
    for name, fieldType in tables.get(schemaTableName, []):
        schemaFieldInfo.append({name: tables[fieldType]} if fieldType in tables else (name, fieldType))
    return schemaTableName, tuple(schemaFieldInfo), structs


def typesCorrespond(fbsType : str, cType : str) -> bool:
    """Check string and vector compatibility before using schema accessors."""
    if ("[" in fbsType) or ("vector" in cType):
        return ("[" in fbsType) and ("vector" in cType)

    if ("string" in fbsType) or ("string" in cType or "char *" in cType):
        return (("string" in fbsType) and ("string" in cType or "char *" in cType))

    return True


def isValidLogType(lines : list) -> bool:
    """Identify concrete logs by their event code and default priority."""
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

def makeTestInfoDict(hppFname : str, baseTypesDict : dict) -> dict:
    """Read a log header and build the context for the Catch2 template."""
    returnInfo = dict()
    with open(hppFname) as headerFile:
        headerLines = headerFile.readlines()

    # add name of test/file/type to be generated
    fNameParts = hppFname.split("/")
    returnInfo["name"] = fNameParts[-1].strip().split(".")[0]
    CamelCase = "".join([word.capitalize() for word in returnInfo["name"].split("_")])
    returnInfo["nameCamelCase"] = CamelCase[0].lower() + CamelCase[1:]
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
        if re.match(r"^\s*(?:explicit\s+)?messageT\s*\(", headerLines[i]):
            messageStructIdxs.append(i)
        if fbMethodName in headerLines[i]:
            hasFbMethods = True

    schemaTableName, schemaFieldInfo, schemaStructs = getSchemaFieldInfo(returnInfo["name"])
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

    returnInfo["messageTypes"] = getMessageFieldInfo(messageStructIdxs, headerLines, schemaFieldInfo, schemaStructs, returnInfo["name"])

    return returnInfo

def getTypeAndName(fieldParts : list) -> tuple[str, str]:
    """Extract a parameter type and name, dropping const and references."""

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

def hasGeneratedHFile(logName : str) -> bool:
    """Check whether this log has a generated FlatBuffers header."""
    generatedFolderPath = "./../types/generated/"
    generatedFolderPath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), generatedFolderPath)
    )

    generatedFilePath = os.path.join(generatedFolderPath, f"{logName}_generated.h")
    if os.path.isfile(generatedFilePath):
        return True

    return False

def getIntSize(type : str) -> int:
    """Return the bit width of a fixed-width integer type, or 32 for int."""
    intSizeBits = 32 # default size 32 bits
    if "_t" in type:
        typeParts = type.split("_t")
        intSizeBits = int(typeParts[0][-1]) if (int(typeParts[0][-1]) == 8) \
                    else int(typeParts[0][-2:])

    return intSizeBits


def getRandInt(type : str) -> int:
    """Choose an integer within the range of its C++ type."""
    unsigned = True if "uint" in type else False

    intSizeBits = getIntSize(type)

    if not unsigned:
        intSizeBits -= 1

    max = (2 ** intSizeBits) - 1
    min = 0 if unsigned else (0 - max - 1)

    return random.randint(min, max)

def getIncrementingInt(type : str) -> int:
    """Advance the deterministic input sequence for an integer type."""
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
    """Produce a C++ scalar initializer, honoring boolean schema conversions."""
    if "bool" in fieldType or (schemaFieldType is not None and "bool" in schemaFieldType):
        return "1"
    elif "string" in fieldType or "char *" in fieldType:
        if gIncrementingVals:
            gNextVals["string"] += 1
            return f'"{gNextVals["string"]}"'
        randString = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
        return f'"{randString}"'
    elif re.fullmatch(r"(?:u?int(?:8|16|32|64)_t|int|unsigned int)", fieldType):
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
    """Generate inputs shared by regular and entropy tests, with valid fixtures."""
    if "fixture" in fieldDict:
        return fieldDict["fixture"]
    if fieldDict["type"] == "timespec":
        return "{123, 456789}"
    if "vector" in fieldDict["type"]:
        if "structMembers" in fieldDict:
            vals = ["{" + ", ".join(getTestValFromType(t) for _, t in fieldDict["structMembers"]) + "}"
                    for i in range(10)]
        else:
            vals = [getTestValFromType(fieldDict["vectorType"]) for i in range(10)]

        # special case telem_pokecenter because vector follows specific format
        if fieldDict["name"] == "pokes" and "vector<float" in fieldDict["type"]:
            catchAssertVals = [vals[i] for i in range(0, len(vals), 2)]
            fieldDict["specialAssertVal"] = f"{{ {",".join(catchAssertVals)} }}"
        return f"{{ {",".join(vals)} }}"

    if "schemaType" in fieldDict:
        return getTestValFromType(fieldDict["type"], fieldDict["schemaType"])

    return getTestValFromType(fieldDict["type"])

def makeFieldAssertions(field : dict, actual : str, expected : str) -> list[str]:
    """Compare serialized components without requiring aggregate operator==."""
    if "structMembers" in field:
        return [f"REQUIRE({actual}.size() == {expected}.size());",
                f"for(size_t i = 0; i < {expected}.size(); ++i)", "{"] + [
                    f"    REQUIRE({actual}[i].{name}() == {expected}[i].{name}());"
                    for name, _ in field["structMembers"]
                ] + ["}"]
    if "schemaMembers" in field:
        return [f"REQUIRE({actual}.{name} == {expected}.{name});" for name, _ in field["schemaMembers"]]
    if "char *" in field["type"]:
        return [f"REQUIRE(strcmp({actual}, {expected}) == 0);"]
    return [f"REQUIRE({actual} == {expected});"]

def findMatchingSchemaField(schemaFieldInfo, fieldName):
    """Find an exact field name in the root or a nested schema table."""
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
    """Record defaults and expand source_location into file and line fields."""
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



def getMessageFieldInfo(messageStructIdxs: list, lines : list, schemaFieldInfo : tuple, schemaStructs : dict, logName : str):
    """Map each constructor parameter to schema fields and test inputs."""
    msgTypesList = []
    # Extract each overload independently.
    for i in range(len(messageStructIdxs)):
        structIdx = messageStructIdxs[i]
        msgsFieldsList = []
        subTableDictIndex = 0
        inMultilineComment = False
        inDefaultArgDef = False

        closed = False
        fieldCount = 0
        while not closed and structIdx < len(lines):

            line = lines[structIdx]
            constructor = re.search(r"\bmessageT\s*\(", line)

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
                       (closeParenCount == openParenCount and constructor):
                        closed = True # parse the field, don't leave loop yet
                        line = line[:line.rfind(")")]

            if inMultilineComment:
                if "*/" in line:
                    inMultilineComment = False
                structIdx += 1
                continue


            # trim line to just get field info
            indexStart = constructor.end() if constructor else 0

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
                    if vectorType in schemaStructs:
                        struct = schemaStructs[vectorType]
                        fieldDict["vectorType"] = struct["type"]
                        fieldDict["type"] = f"std::vector<{struct['type']}>"
                        fieldDict["structMembers"] = struct["members"]

                if fieldType == "timespec":
                    # A timestamp occupies consecutive seconds/nanoseconds fields.
                    fieldDict["schemaMembers"] = list(zip(
                        ("tv_sec", "tv_nsec"),
                        (schemaFieldInfo[fieldCount][0], schemaFieldInfo[fieldCount + 1][0])
                    ))
                    fieldCount += 2
                elif len(schemaFieldInfo) != 0:

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

                if name in gTestFixtures.get(logName, {}):
                    fieldDict["fixture"] = gTestFixtures[logName][name]
                fieldDict["testVal"] = makeTestVal(fieldDict)

                # add field dict to list of fields
                msgsFieldsList.append(fieldDict)

                # note we do this after the fieldDict has been appended to msgsFieldList.
                # This is because the function setDefaultArgOfLastField() does some
                # special casing to replace std::source_location with file and line
                # within the msgsFieldsList
                if "=" in field:
                    setDefaultArgOfLastField(msgsFieldsList, field)

            structIdx += 1

        msgTypesList.append(msgsFieldsList)

    return msgTypesList

def makeInheritedTypeInfoDict(typesFolderPath : str, baseName : str, logName : str) -> dict:
    """Build a test context using the inherited message constructor."""
    returnInfo = dict()

    baseFilePath = os.path.join(typesFolderPath, f"{baseName}.hpp")
    with open(baseFilePath) as baseHFile:
        baseHLines = baseHFile.readlines()

    # add name of test/file/type to be generated
    returnInfo["name"] = logName
    returnInfo["genTestFname"] = f"{returnInfo['name']}_generated_tests.cpp"
    returnInfo["className"] = "C" + "".join([word.capitalize() for word in returnInfo["name"].split("_")])
    CamelCase = "".join([word.capitalize() for word in returnInfo["name"].split("_")])
    returnInfo["nameCamelCase"] = CamelCase[0].lower() + CamelCase[1:]
    returnInfo["classVarName"] = "".join([word[0].lower() for word in returnInfo["name"].split("_")])
    returnInfo["baseType"] = baseName
    returnInfo["hasGeneratedHfile"] = hasGeneratedHFile(logName)

    # find where messageT structs are being made in base log file -> describes fields
    messageStructIdxs = []
    for i in range(len(baseHLines)):
        if re.match(r"^\s*(?:explicit\s+)?messageT\s*\(", baseHLines[i]):
            messageStructIdxs.append(i)

    schemaTableName, schemaFieldInfo, schemaStructs = getSchemaFieldInfo(baseName)

    returnInfo["schemaTableName"] = schemaTableName
    msgFieldInfo = getMessageFieldInfo(messageStructIdxs, baseHLines, schemaFieldInfo, schemaStructs, logName)

    returnInfo["messageTypes"] = [[]] if "empty_log" in baseName else msgFieldInfo

    return returnInfo

def versionAsNumber(major, minor):
    """Encode a Python major/minor pair for version comparisons."""
    return (major * 1000 + minor)

def main():
    """Parse generator options and render regular and inherited log tests."""
    # The generator uses Python 3.12 f-string syntax.
    if (versionAsNumber(sys.version_info[0], sys.version_info[1]) < versionAsNumber(3,12)):
        print("Error: Python version must be >= 3.12")
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
    env.globals["makeFieldAssertions"] = makeFieldAssertions
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
