import idapro
import idautils
import idc
print(dir(idc))

idapro.open_database("temp.exe", True)

for index, func in enumerate(idautils.Functions()):
    func_name = idc.get_func_name(func)
    print(f"{index}] Function Name: {func_name}, Function Address: 0x{func:X}")
    start_ea = idc.get_func_attr(func, idc.FUNCATTR_START)
    end_ea = idc.find_func_end(func)
    for head in idautils.Heads(start_ea, end_ea):
        if idc.is_code(idc.get_full_flags(head)):
            print(f"{hex(head)}: {idc.GetDisasm(head)}")

idapro.close_database(save=False)