import os
import tempfile

# The DDS trace sink must be per-user: a hardcoded /tmp/cdds.LOG is created by
# whichever account runs first on a shared machine, and every other user then
# fails domain creation with "cannot open for writing".
ChannelTraceOutputFile = os.path.join(
    tempfile.gettempdir(), f"cdds_{os.getuid()}.LOG"
)

ChannelConfigHasInterface = '''<?xml version="1.0" encoding="UTF-8" ?>
    <CycloneDDS>
        <Domain Id="any">
            <General>
                <Interfaces>
                    <NetworkInterface name="$__IF_NAME__$" priority="default" multicast="default"/>
                </Interfaces>
            </General>
            <Tracing>
                <Verbosity>config</Verbosity>
            <OutputFile>$__TRACE_FILE__$</OutputFile>
        </Tracing>
        </Domain>
    </CycloneDDS>'''

ChannelConfigAutoDetermine = '''<?xml version="1.0" encoding="UTF-8" ?>
    <CycloneDDS>
        <Domain Id="any">
            <General>
                <Interfaces>
                    <NetworkInterface autodetermine=\"true\" priority=\"default\" multicast=\"default\" />
                </Interfaces>
            </General>
        </Domain>
    </CycloneDDS>'''
