# State Rule Engine

This document describes the configuration and operation of the state rule engine, `stateRuleEngine`, which is designed to provide a system for evaluating the state of MagAO-X to provide user feedback when the system is not in the correct state for observations.

## Example
Suppose we want to caution the user when the FPM wheel is in a certain position and the focus stage is not in the corresponding position to produce an in-focus image. Recall that information about the instrument is stored in INDI properties, with the logical structure `device.property.element=value`.  We can set up a "rule" to notify the user of this out-of-focus state by creating comparisons between such properties as follows.

First we create a rule to test if `fwfpm` is in the `fpm` position:
```toml
#This rule tests if fwfpm is in the fpm position
[fwfpm-fpm]
ruleType=swVal
priority=none #means don't publish, is the default and can be left out
comp=Eq #the default, can be left out
property=fwfpm.filterName #specifies device.property-name
element=fpm #specifies the element within property
target=On #the value to compare to
```

This rule will evaluate to `true` whenever the "switch" property element `fwfpm.filtername.fpm` is in state `On`.  This shows that a rule is named by its TOML section heading, and rules have types specified by the `ruleType` keyword.  The type specifies what is being compared.  The comparison to use is specified by `comp` and only some comparisons are valid for a given rule type.  See below for more.

One detail to keep track of is that a device like `fwfpm` will report its
position while still moving. So next we create a rule to check if `fwfpm` is moving or not.  We can do this with the `fwfpm.fsm_state` property and its `state` element, which will be `READY` if `fwfpm` is
stopped in its commanded position:

```toml
#This rule tests if fwfpm is in state READY
[fwfpm-READY]
ruleType=txtVal
property=fwfpm.fsm_state
element=state
target=READY
```
The above rule will be `true` if and only if `fwfpm.fsm_state.state==READY`.

Now we combine these into a rule to test if `fwfpm` is stopped 
in the `fpm` position:

```toml
#This tests if fwfpm is in fpm in state READY (not moving)
[fwfpm-fpm-READY]
ruleType=ruleComp
comp=And
rule1=fwfpm-READY
rule2=fwfpm-fpm
```

The above rule performs the logical AND comparison between two rules, and so is `true` if and only if both are `true`.

One more rule we need is a test if the two devices `fwfpm` and `stagesci1` are not in the corresponding position. 
```toml
#this rule tests if the preset names for fwfpm and stagesci1 are different
[fwfpm-stagesci1-neq]
ruleType=elCompSw
property1=fwfpm.filterName
element1=fpm
property2=stagesci1.presetName
element2=fpm
comp=Neq
```
The above rule will evaluate to `true` if `fwfpm.filterName.fpm` and `stagesci1.presetName.fpm` are not in the same position by comparing the states of their corresponding preset/filter selection switches.

Finally, we combine all of the above rules into a rule to raise a caution if `fwfpm` is in position `fpm`, stopped, and `stagesci`
is not in the fpm position.
```toml
#This rule raises a caution if fwfpm is in fpm & READY, and stagesci1 is not in fpm
[fwfpm-fpm-stagesci-fpm]
ruleType=ruleComp
priority=caution
message=fwfpm is in fpm but stagesci1 is not in focus position fpm
rule1=fwfpm-fpm-READY
rule2=fwfpm-stagesci1-neq
comp=And
```
Now the user can be notified to take caution whenever this out-of-focus state occurs.  The value of the `message` keyword is used for notifications.

## Rule Configuration

The rules are configured using the usual MagAO-X TOML .conf files.  A rule is named by its TOML section heading, e.g. `[rule-name]`, and then 
specified by the keywords.  Which keywords are valid depends on the ruleType.  The following table lists the keywords.

| keyword     | Required | Default | ruleTypes  | Purpose                    | 
|:-----------:|:--------:|:-------:|------------|--------------------|
| ruleType    | Y        |         |            | the type of rule |
| priority    | N        | none    | all        | the reporting priority |
| message     | N        |         | all        | descriptive message used for user feedback |
| comp        | N        | Eq      | all        | the comparison to use |
| property    | Y        |         | numVal, txtVal, swVal | the INDI property |
| element     | Y        |         | numVal, txtVal, swVal | the element within property |
| property1   | Y        |         | elCompNum, elCompTxt, elCompSw | the first INDI property |
| element1    | Y        |         | elCompNum, elCompTxt, elCompSw | the element within property1 |
| property2   | Y        |         | elCompNum, elCompTxt, elCompSw | the second INDI property |
| element2    | Y        |         | elCompNum, elCompTxt, elCompSw | the element within property2 |
| rule1       | Y        |         | ruleComp   | the first rule |
| rule2       | Y        |         | ruleComp   | the second rule |
| tol         | N        | 1e-6    | numVal, elCompNum | the tolerance for equality of numbers |

Note that required keywords are only required in their respective ruleTypes. 

Additional information about keywords is provided below.

### Keywords ruleType and comp

The `ruleType` keyword specifies the type of rule, and the `comp` keyword specifies the logical comparison to use.  They go together as described in the following table.

`ruleType`  | Description                                                    |  Valid comparisons for `comp` | Notes
|:---------:|----------------------------------------------------------------|:-----------------------------:|-----------------------------|
numVal      | compare the value of a number element to a numeric value | Eq, Neq, Lt, LtEq, Gt, GtEq          | equality is tested with a tolerance|
txtVal      | compare the value of a text element to a text value     | Eq, Neq                            ||
swVal       | compare the state of a switch to either `On` or `Off`   | Eq, Neq                            ||
elCompNum   | compare the value of two number elements to each other    | Eq, Neq, Lt, LtEq, Gt, GtEq          | equality is tested with a tolerance|
elCompTxt   | compare the value of two text elements to each other      | Eq, Neq                            ||
elCompSw    | compare the value of two switch elements to each other    | Eq, Neq                            ||
ruleComp    | compare the value of two rules to each other                   | Eq/Xnor, Neq/Xor, And, Nand, Or, Nor ||


### Keyword tol
For numerical comparisons, equality is tested with a tolerance, specified by the keyword `tol` in the config file.  This accounts for floating point nonsense and the binary-text-binary conversions inherent in the INDI protocol.  The default for `tol` is `1e-6`.  If you set it to 0 you will get strict equality checking.

## Future Plans

- [ ] compare attributes, e.g. timestamp
- [ ] add support for lights for completeness
- [ ] add a "which switch in a SwitchVector" is on rule 