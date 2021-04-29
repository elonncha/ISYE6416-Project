import numpy as np
import pandas as pd
from zipfile import ZipFile

def load_ejscreen(data_path = 'data/ejscreen/EJSCREEN_2020_USPR.csv',
                  meta_path = 'data/ejscreen/ejscreen_meta.xlsx',
                  unzip = True):
    '''

    :param data_path:
    :param meta_path:
    :param unzip:
    :return:
    '''
    if unzip:
        with ZipFile('data/ejscreen/EJSCREEN_2020_USPR.csv.zip', 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall('data/ejscreen')


    ejscreen, meta = pd.read_csv(data_path), pd.read_excel(meta_path)

    # column filtering
    col_mask = np.isin(ejscreen.columns.values, meta['Name'])
    ejscreen = ejscreen.loc[:,col_mask]

    # na drop
    for c in ejscreen.columns.values: # coerce type casting
        if c != 'ID':
            ejscreen[c] = pd.to_numeric(ejscreen[c], errors = 'coerce')
    ejscreen = ejscreen.dropna()

    # fix block id
    ejscreen['ID'] = ejscreen['ID'] / 10
    ejscreen['ID'] = ejscreen['ID'].astype('int')
    ejscreen['ID'] = ejscreen['ID'].astype('str')

    block_id = []
    for id in ejscreen['ID']:
        if len(id) == 10:
            block_id.append('0' + id)
        else:
            block_id.append(id)

    ejscreen['ID'] = block_id

    # aggregate block to tract by weight (population)
    sum_agg = ejscreen.groupby('ID').agg({'ACSTOTPOP':'sum',
                                          'AREALAND':'sum',
                                          'AREAWATER':'sum',
                                          'NPL_CNT':'sum',
                                          'TSDF_CNT':'sum'})

    tract_pop = ejscreen.groupby('ID')['ACSTOTPOP'].agg('sum')
    mean_agg = pd.merge(ejscreen.drop(columns = ['AREALAND','AREAWATER', 'NPL_CNT', 'TSDF_CNT']), tract_pop,
                        how = 'left', left_on='ID', right_on=tract_pop.index)
    mean_agg['ACSTOTPOP_y'] = mean_agg['ACSTOTPOP_x'] / mean_agg['ACSTOTPOP_y'] # weight

    for c in mean_agg.columns.values:
        if c not in ['ID','ACSTOTPOP_x', 'ACSTOTPOP_y']:
            mean_agg[c] = mean_agg[c] * mean_agg['ACSTOTPOP_y'] # multiply block value by weight

    mean_agg = mean_agg.groupby('ID').agg('mean').drop(columns = ['ACSTOTPOP_x', 'ACSTOTPOP_y']) # aggregate

    ejs_df = sum_agg.join(mean_agg, how = 'inner', on = sum_agg.index).dropna()

    return(ejs_df)


def load_cdc():
    pass


def combining_data():
    ejs = load_ejscreen(unzip = True)
    cdc = pd.read_csv('data/cdc/cdc_places.csv')
    # cdc = load_cdc()

    # fix tract id
    cdc['TractFIPS'] = cdc['TractFIPS'].astype('int')
    cdc['TractFIPS'] = cdc['TractFIPS'].astype('str')
    tract_id = []
    for id in cdc['TractFIPS']:
        if len(id) == 10:
            tract_id.append('0' + id)
        else:
            tract_id.append(id)

    cdc['TractFIPS'] = tract_id

    # na drop
    for c in cdc.columns.values:  # coerce type casting
        if c != 'TractFIPS':
            cdc[c] = pd.to_numeric(cdc[c], errors='coerce')
    cdc = cdc.dropna()

    # divide percent by 100
    cdc.iloc[:,1:] = cdc.iloc[:,1:] / 100

    # join
    data = ejs.merge(cdc, how = 'left', left_on = 'key_0', right_on = 'TractFIPS')
    data_cleaned = data.drop(columns = ['key_0', 'AREALAND', 'AREAWATER', 'TractFIPS']).dropna()

    #output
    data_cleaned.to_csv('data/data_cleaned.csv')


if __name__ == '__main__':
    combining_data()