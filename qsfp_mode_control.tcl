# Persistent, local control bridge. No FPGA programming and no timed RX control.
lassign $argv probes_file ready_file owner_pid
open_hw_manager
connect_hw_server -url TCP:localhost:3121
set targets [get_hw_targets -quiet *210249AE83E1*]
if {[llength $targets]!=1} {error "Expected one HTG-930 target"}
open_hw_target [lindex $targets 0]
set device [lindex [get_hw_devices -filter {PART =~ *xcvu9p*}] 0]
if {$device eq ""} {error "xcvu9p not found"}
current_hw_device $device
set_property PROBES.FILE $probes_file $device
refresh_hw_device $device
proc probe {name} {
    set p [get_hw_probes -quiet $name]
    if {[llength $p]!=1} {error "Missing $name; load the controller mode bitstream once"}
    return $p
}
set pstatus [probe startup_status]
set pconfig [probe startup_config]
set vio [get_hw_vios [get_property HW_VIO $pstatus]]
set mgmt_in [probe mgmt_status]
set mgmt_out [probe mgmt_command]
set select_out [probe module_select_n]
set mgmt_vio [get_hw_vios [get_property HW_VIO $mgmt_in]]
foreach p [list $pstatus $mgmt_in] {set_property INPUT_VALUE_RADIX HEX $p}
foreach p [list $pconfig $mgmt_out $select_out] {set_property OUTPUT_VALUE_RADIX HEX $p}
refresh_hw_vio -update_output_values $vio
refresh_hw_vio -update_output_values $mgmt_vio
proc number {p prop} {return [expr 0x[get_property $prop $p]]}
set toggle [expr {([number $pconfig OUTPUT_VALUE]>>32)&1}]
set ticks 0
set mode off
proc control {abort clear} {
    global toggle ticks pconfig vio
    set value [expr {$ticks|($toggle<<32)|($abort<<33)|($clear<<34)}]
    set_property OUTPUT_VALUE [format %016X $value] $pconfig
    commit_hw_vio $vio
}
proc snapshot {} {
    global vio pstatus
    refresh_hw_vio $vio
    set s [number $pstatus INPUT_VALUE]
    if {($s>>224)!=0x42555231} {error "Controller-mode image signature mismatch"}
    return $s
}
proc select_module {port} {
    global select_out mgmt_vio
    set_property OUTPUT_VALUE [format %X [expr {$port==3 ? 2 : 1}]] $select_out
    commit_hw_vio $mgmt_vio
}
proc i2c {read reg data} {
    global mgmt_in mgmt_out mgmt_vio
    refresh_hw_vio $mgmt_vio
    set before [number $mgmt_in INPUT_VALUE]
    if {$before&1} {error "I2C engine busy"}
    set old [number $mgmt_out OUTPUT_VALUE]
    set cmd [expr {($data<<32)|(($old^0x80000000)&0x80000000)|($read ? 0x01405000 : 0x0040D000)|$reg}]
    set_property OUTPUT_VALUE [format %010X $cmd] $mgmt_out
    commit_hw_vio $mgmt_vio
    set deadline [expr {[clock milliseconds]+5000}]
    while {[clock milliseconds]<$deadline} {
        refresh_hw_vio $mgmt_vio
        set s [number $mgmt_in INPUT_VALUE]
        if {(($s>>16)&65535)!=(($before>>16)&65535) && ($s&2)} {
            if {($s&0xF6)!=6} {error [format "I2C transaction failed: %08X" $s]}
            return [expr {($s>>8)&255}]
        }
        after 2
    }
    error "I2C transaction timed out"
}
proc laser {value} {
    select_module 3
    i2c 0 86 $value
    if {[i2c 1 86 0]!=$value} {error "TX_DISABLE readback mismatch"}
}
proc dark {} {control 1 0; laser 15}
proc wait_flag {mask value timeout_ms} {
    set end [expr {[clock milliseconds]+$timeout_ms}]
    while {[clock milliseconds]<$end} {
        set s [snapshot]
        if {($s&$mask)==$value} {return $s}
        after 2
    }
    error [format "FPGA state wait timed out: status=%064llX" $s]
}
# Check the signature before issuing any output change.
snapshot
# Leave the reverse transmitter off. RX state remains entirely receiver-local.
select_module 4
i2c 0 86 15
if {[i2c 1 86 0]!=15} {error "QSFP4 TX_DISABLE readback mismatch"}
dark
set finished 0
set client {}
proc handle {channel} {
    global mode ticks toggle finished client
    if {[eof $channel]} {
        catch {dark}; catch {close $channel}; set finished 1; return
    }
    if {[gets $channel line]<0} {return}
    set failed [catch {
        if {[llength $line]>3} {error "Invalid control command"}
        lassign $line op arg n
        switch -- $op {
            idle {
                if {$arg ni {off burst continuous}} {error "Invalid mode"}
                control 1 0
                set mode $arg
                if {$mode eq "continuous"} {
                    laser 13
                    # Host loopback readiness check only; no signal is sent to RX.
                    wait_flag 30 30 10000
                } else {laser 15}
            }
            prepare {
                if {$arg ni {burst continuous} || ![string is integer -strict $n] || $n<0 || $n>200000000} {error "Invalid mode/guard"}
                set mode $arg; set ticks $n
                control 1 0
                if {$mode eq "burst"} {laser 15} else {laser 13; wait_flag 30 30 10000}
                control 1 1
                after 2
                control 1 0
                # FIFO busy and empty flags: same criterion as the sweep runner.
                wait_flag [expr {255<<42}] [expr {0xC2<<42}] 5000
                control 0 0
            }
            start {
                # The first complete H2C packet must be in the hardware FIFO.
                wait_flag [expr {1<<41}] [expr {1<<41}] 5000
                set toggle [expr {1-$toggle}]
                control 0 0
                set s [snapshot]
                if {($s>>51)&1} {error "FPGA rejected payload-gate arm"}
                # Successful target-write STOP starts the hardware guard timer.
                laser 13
                wait_flag [expr {1<<38}] [expr {1<<38}] 5000
            }
            finish {
                if {$arg ni {burst continuous}} {error "Invalid mode"}
                if {$arg eq "burst"} {dark}
            }
            status {}
            times {}
            quit {dark; set finished 1}
            default {error "Unknown control command"}
        }
        set value [snapshot]
        if {$op eq "times"} {set value [number [probe startup_times] INPUT_VALUE]}
        set response [format "OK %064llX" $value]
    } message]
    if {$failed} {
        catch {dark}
        set response "ERR [binary encode base64 -maxlen 0 [encoding convertto utf-8 $message]]"
    }
    if {[catch {puts $channel $response; flush $channel}]} {catch {dark}; set finished 1}
}
proc accept {channel address port} {
    global client
    if {$client ne ""} {close $channel; return}
    set client $channel
    fconfigure $channel -blocking 0 -buffering line -encoding ascii
    fileevent $channel readable [list handle $channel]
}
proc owner_watchdog {} {
    global owner_pid finished
    if {[catch {exec kill -0 $owner_pid}]} {catch {dark}; set finished 1; return}
    after 1000 owner_watchdog
}
set server [socket -server accept -myaddr 127.0.0.1 0]
set port [lindex [fconfigure $server -sockname] 2]
set f [open ${ready_file}.tmp w]; puts $f $port; close $f
file rename ${ready_file}.tmp $ready_file
after 1000 owner_watchdog
vwait finished
catch {dark}
catch {close $server}
if {$client ne ""} {catch {close $client}}
close_hw_manager
exit
