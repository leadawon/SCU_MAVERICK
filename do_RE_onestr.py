import generation.legacy_but_important.version_2.atomic_amr_short as aasc

stog, gtos = aasc.load_models('/workspace/SCU_MAVERICK')
facts = aasc.process_summary("The co-operative group has said The co-operative group is unlikely to declare any dividends before 2018.", stog, gtos)
print(facts)