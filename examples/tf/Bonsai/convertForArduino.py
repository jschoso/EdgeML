import os
def hex_to_c_array(hex_data, var_name):

    c_str = ''

    # Create header guard
    c_str += '#indef' + var_name.upper() + '_H\n'
    c_str += '#define' + var_name.upper() + '_H\n'

    # Add array length at top of file
    c_str += '\unsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'

    # Declare C variable
    c_str += 'unsigned char ' + var_name + '[] = {'
    hex_array = []
    
    for i, val in enumerate(hex_data) :

        # Construct string from hex
        hex_str = format(val, '#04x')

        # Add formatting so each line stays within 80 characters
        if (i + 1) < len(hex_data):
            hex_str += ','
        if (i + 1) % 12 == 0:
            hex_str += '\n '
        hex_array.append(hex_str)

        # Add closing brace
        c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'

        # Close out header guard
        c_str += '#endif //' + var_name.upper() + '_H'

        return c_str
    
# Datei einlesen und hex_data erstellen
file_path = '/home/jschosto/EdgeML/examples/tf/Bonsai/usps10/TFBonsaiResults/02_10_27_18_11_24/bonsai_model.tflite'
var_name = 'bonsai_model'

if os.path.exists(file_path):
    with open(file_path, 'rb') as f:
        hex_data = f.read()  # Dateiinhalt in Binärdaten lesen

    # C-Header-Code generieren
    c_header = hex_to_c_array(hex_data, var_name)

    # C-Header-Datei speichern
    output_file = f'{var_name}.h'
    with open(output_file, 'w') as out_file:
        out_file.write(c_header)

    print(f'C-Header-Datei wurde erfolgreich generiert: {output_file}')
else:
    print(f'Datei nicht gefunden: {file_path}')