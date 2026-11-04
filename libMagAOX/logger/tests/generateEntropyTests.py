#!/bin/env python3

'''Generate mixed-log round-trip tests using the regular generator's fixtures.

Options select the number of log types (-n), repetitions (-e), seed (-s),
and required comma-separated log types (-f). See flatlogs.md for usage.
'''

import os
import sys
import pathlib
import getopt
import random
from generateTemplatedCatch2Tests import *
import json

# check jinja2 is installed. install it if not
try:
    import jinja2
except ModuleNotFoundError:
    print("module 'Jinja2' is not installed. Installing Jinja2...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'Jinja2'])
    import jinja2


def usage():
    """Print accepted command-line options and exit."""
    print("Usage: python3 ./generateEntropyTests.py\n" +
          "-n <number of types>\n" +
          "-e <entropy>\n" +
          "-s <random seed>\n" +
          "-f <flatlog types to use>")
    exit(0)

def main():
    """Select log types, generate inputs and assertions, and render the test."""
    # default entropy, number of flatlog types, and random seed
    seed = 1
    entropy = 1
    nTypes = 2
    desiredTypes = []

    # path to .hpp files
    typesFolderPath = "./../types"
    typesFolderPath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), typesFolderPath)
    )

    # path to gen tests
    genTestsFolderPath = "./generated_tests"
    genTestsFolderPath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), genTestsFolderPath)
    )

    # get list of flatlogs for which there are generated_tests
    allTypes = [name for name in os.listdir(genTestsFolderPath) if name.endswith("_generated_tests.cpp")]
    if (nTypes > len(allTypes)):
        print(f"Error: n cannot be larger than amount of types in generated_tests. Retry with n < {len(allTypes)}.")
        exit(0)
    allTypes.sort()

    # get opt
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:e:s:f:")
    except getopt.GetoptError:
        usage()

    for opt, arg in opts:
        if opt in ["-e"]:
            if not arg.isdigit():
                print(f"Error: entropy {arg} provided is not an integer.")
                exit(0)
            entropy = int(arg)
        elif opt in ["-n"]:
            if not arg.isdigit():
                print(f"Error: number {arg} provided is not an integer.")
                exit(0)
            nTypes = int(arg)
        elif opt in ["-s"]:
            if not arg.isdigit():
                print(f"Error: random seed {arg} provided is not an integer.")
                exit(0)
            # use random seed if provided with -s
            seed = int(arg)
        elif opt in ["-f"]:
            if arg.strip() != "":
                desiredTypes = [f"{x.strip()}_generated_tests.cpp" for x in arg.split(",")]

            # check generated test exists for desired flatlog types
            for dType in desiredTypes:
                if dType not in allTypes:
                    print(f"Error: there is not a generated test type for requested type '{dType[:dType.index("_generated_tests.cpp")]}'")
                    exit(0)


    random.seed(seed)

    # load template
    env = jinja2.Environment(
        loader = jinja2.FileSystemLoader(searchpath="./")
    )
    env.trim_blocks = True
    env.lstrip_blocks = True
    testTemplate = env.get_template("entropyTestTemplate.jinja2")



    # use required types
    if nTypes < len(desiredTypes):
        print(f"Error: n={nTypes} is less than desired flatlog types: {desiredTypes}. Please select n >= number of desired types.")
        exit(0)

    # check desired types are in allTypes and add them
    testTypes = []
    for dType in desiredTypes:
        assert(dType in allTypes)  # sanity check, validated before this
        testTypes.append(dType)
        allTypes.remove(dType)

    # randomly select remaining nTypes
    for _ in range(nTypes - len(desiredTypes)):

        randomIdx = random.randint(0, len(allTypes) - 1)
        testTypes.append(allTypes[randomIdx])

        del allTypes[randomIdx]


    # construct info dictionary for each type
    baseTypesDict = dict()
    typesInfoList = []
    for type in testTypes:

        # check valid type to generate info for
        if "_generated_tests.cpp" not in type:
            continue
        typeName = type[:type.index("_generated_tests.cpp")]
        typePath = os.path.join(typesFolderPath, f"{typeName}.hpp")

        # make dictionary with info for template
        info = makeTestInfoDict(typePath, baseTypesDict)

        if info is None:
            # check if this is an inherited type
            for baseType, inheritedTypes in baseTypesDict.items():
                for inheritedType in inheritedTypes:
                    if inheritedType in type:
                        info = makeInheritedTypeInfoDict(typesFolderPath, baseType, inheritedType)

        assert(info is not None)
        typesInfoList.append(info)

    # make big string of const fields & big string of asserts in tandem
    objCount = 0
    objectCtors = []
    testVariables = []
    catchAsserts   = []
    totalFieldCount = 0
    for _ in range(entropy):

        for type in typesInfoList:

            objName    = f"{type["classVarName"]}_{objCount}"
            arguments = []

            for field in type["messageTypes"][0]:
                if "defaultArg" in field:
                    # Defaults (including the wrapper's source_location) are
                    # checked in the regular generated scenario.
                    continue

                # make test variable
                testValName = f"{type["classVarName"]}{field["name"].capitalize()}_{str(totalFieldCount).rjust(8, '0')}"

                arguments.append(f"(char *) {testValName}" if "char *" in field["type"] else testValName)

                constVarStr = f"const {field["type"]} {testValName} = {makeTestVal(field)};"
                testVariables.append(constVarStr)

                # Use the same transformed-value and component comparisons as regular tests.
                expected = testValName
                if "specialAssertVal" in field:
                    expected += "_expected"
                    testVariables.append(f"const {field['type']} {expected} = {field['specialAssertVal']};")
                catchAsserts.extend(makeFieldAssertions(field, f"{objName}.m_{field['name']}", expected))

                totalFieldCount += 1

            catchAsserts.append(f"REQUIRE({objName}.m_verify);")
            objectCtors.append(f"{type['className']}_0 {objName}({', '.join(arguments)});" if arguments
                               else f"{type['className']}_0 {objName};")
            objCount += 1

    jinjaDict = dict()
    jinjaDict["types"]         = typesInfoList
    jinjaDict["seedOpt"]       = seed
    jinjaDict["dTypesOpt"]     = desiredTypes
    jinjaDict["nTypesOpt"]     = nTypes
    jinjaDict["entropyOpt"]    = entropy
    jinjaDict["objectCtors"]   = objectCtors
    jinjaDict["catchAsserts"]  = catchAsserts
    jinjaDict["testVariables"] = testVariables


    # print(json.dumps(jinjaDict, indent=4))

    # render
    renderedTest = testTemplate.render(jinjaDict)

    # generated tests output path
    outFolderPath = "./gen_entropy_tests/"
    outFolderPath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), outFolderPath)
    )

    # make directory if it doesn't exist
    pathlib.Path(outFolderPath).mkdir(exist_ok=True)

    # write out file
    outFilename = f"generated_test_e{entropy}_n{nTypes}.cpp"
    outPath = os.path.abspath(
        os.path.join(outFolderPath, outFilename)
    )
    with open(outPath,"w") as outfile:
        print(renderedTest,file=outfile)




if (__name__ == "__main__"):
    main()

