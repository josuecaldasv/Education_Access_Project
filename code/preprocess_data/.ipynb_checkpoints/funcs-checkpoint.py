
import pandas as pd
import numpy as np
from glob import glob
import variables as vb
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer


# 1. IMPORTACIÓN Y PREPROCESAMIENTO
###################################

def importar_bases( ruta ):
    '''
    Propósito:
        - Abre multiples datasets al mismo tiempo
    Inputs:
        - La ruta de acceso donde se encuentra las datasets
    Outputs:
        - La función crea una lista de bases indexada.
          Para acceder a acada una de las bases solo hay que 
          llamar a la lista con el índice correspondiente.
          Por ejemplo: lista_bases[0]
    Especificaciones:
        - Las datasets se encuentran en formato .dta
        - Si que quiere abrir todas las datasets de una determinada
          carpeta, se adiciona la siguiente expresión a la ruta de
          acceso de la carpeta "*.dta". 
    '''
    ruta_de_acceso = glob( ruta )
    lista_bases = []
    for i, base in enumerate( ruta_de_acceso ):
        dataframes = pd.read_stata( base,
                                    convert_categoricals = False,
                                    convert_dates = False,
                                    convert_missing = True )
        lista_bases.append( dataframes )
    return lista_bases


# 2. FILTROS: SELECCIÓN DE VARIABLES
####################################
        
def filtro_missings( dataframes, umbral ):
    '''
    Propósito: 
        - Aplicar un filtro de variables basado en el porcentaje
          de valores perdidos de cada columna. Al aplicar el filtro,
          permanecen las columnas con un porcentaje de valores perdidos
          menor al umbral asignado.
    Inputs:
        - dataframes: Lista de bases de datos
        - umbral: porcentaje de valores perdidos
    Output:
        - Base de datos filtrada de variables con valores perdidos igual o mayor
          al umbral espeficiado.
    '''
    for i, dataframe in enumerate( dataframes ):
        dataframes[i] = dataframes[i].loc[ : , dataframes[i].isnull().mean() < umbral ]
    return dataframes


def filtro_variabilidad( datasets, umbral ):
    '''
    Propósito:
        - Aplicar un filtro de columnas (variables) basado en la variabilidad
          de las columnas. Al aplicar el filtro, permanecen las columnas con
          un porcentaje de variabilidad mayor al umbral asignado.
    Inputs:
        - datasets: Lista de bases de datos
        - umbral: umbral de variabilidad
    Output:
        - Base de datos filtrada de columnas con una variabilidad mayor o igual al
          umbral asignado.
    Especificaciones:
        - Esta función utiliza el algoritmo: sklearn.feature_selection.VarianceThreshold
        - La implementación del algoritmo se basa en el siguiente texto: 
          https://medium.com/nerd-for-tech/removing-constant-variables-feature-selection-463e2d6a30d9
    '''
    for i, dataset in enumerate( datasets ):
        var_thr = VarianceThreshold( threshold = umbral )
        var_thr.fit( datasets[i] )
        var_thr.get_support()
        columnas = [ col for col in datasets[i].columns if col not in datasets[i].columns[ var_thr.get_support() ] and col not in vb.dep_vars ]
        datasets[i].drop( columnas, axis = 1, inplace = True )
    return datasets

        

def filtro_correlacion( dataset, dep, umbral ):
    '''
    Propósito:
        - Aplicar un filtro de columnas (variables) basado en la correlación
          de columnas. Si la correlación entre dos variables predictoras supera
          el umbral, se elimina la variable que tiene menor correlación con la 
          variable dependiente.
    Inputs:
        - dataset: base de datos
        - dep: variable dependiente. Usualmente se llama así: 
          dataset['dep'].reset_index()
          umbral: umbral de correlación
    Output:
        - Base de datos en la que se eliminaron las variables descritas en el
          Propósito.
    Especificaciones:
        - A diferencia de las anteriores funciones, en este caso la función
          trabaja sobre una base de datos. El filtro debe repetirse por cada
          base de datos y por cada variable dependiente.
        - La función "corr" (calcula correlaciones) ignora por default aquellas
          filas con valores perdidos. 
    Créditos:
        - Basado en la función "low_corr_vars" creada por el profesor A. Cozzubo
          en: https://github.com/acozzubo/ML_Public_Policy
    '''
    candidatas = dataset.columns.values.tolist()
    for var1 in dataset.columns.values.tolist():
        for var2 in dataset.columns.values.tolist():
            if var1 != var2 and var1 in candidatas and var2 in candidatas:
                if abs( dataset[var1].corr( dataset[var2] ) ) > umbral:
                    if abs( dataset[var1].corr( dep ) ) <= abs( dataset[var2].corr( dep ) ):
                        candidatas.remove( var1 )
                    else:
                        candidatas.remove( var2 )
    return dataset[candidatas]



# 3. MÉTODOS DE IMPUTACIÓN
##########################
        
def imputar_i( dataframes, vars, val, dummy = True ):
    '''  
    Propósito:
       -  Imputar con un valor asignado por el usuario una lista determinada
          de variables, con la posibilidad de generar variables dummies de
          control por cada variable imputada.
    Inputs:
        - dataframes: lista de dataframes a imputar
        - vars: lista de variables a imputar
        - val: valor que se usará para imputar
        - dummy: booleano. True en caso se desee generar una variable dummy 
          de control por cada variable imputada. 
    Output:
        - Lista de base de datos imputada en las columnas seleccionadas, 
          con el valor especificado, y con/sin variables dummy de control.
    Créditos:
        - Basado en la función "mv_treat" creada por el profesor A. Cozzubo
          en: https://github.com/acozzubo/ML_Public_Policy
    '''
    for dataframe in dataframes:
        for var in vars:
            if var in dataframe.columns:
                if dataframe[var].isnull().sum() > 0:                
                    if dummy:
                        dataframe[ 'im_' + var ] = 0
                        dataframe.loc[ dataframe[var].isnull(), 'im_' + var ] = 1                
                    dataframe.loc[ dataframe[var].isnull(), var ] = val
    return dataframes



def imputar_ii( dataframe, num = True, dummy = True ):
    '''
    Propósito:
        - Imputar con media o moda las bases de datos de la lista
          de bases brindada, con la posibilidad de generar variables
          dummies de control por cada variable imputada.
    Inputs:
        - dataframes: lista de bases de datos
        - siaf: booleano. Si es True, se imputa las variables de SIAF
          con media. Si es False, se imputa las variables de Renamu con
          moda.
        - dummy: booleano. Si es True, se generan variables de control 
          por cada una de las variables imputadas.
    Output:
        - Lista de bases de datos imputada en el conjunto de variables 
          seleccionadas, sean provenientes de Renamu o SIAF, con el valor 
          especificado, y con/sin variables dummy de control.
    '''
    for var in dataframe.columns:
        if dataframe[var].isna().sum() > 0:
            if num:
                if var in vb.num_vars:
                    if dummy:
                        dataframe[ 'im_' + var ] = 0
                        dataframe.loc[ dataframe[var].isnull(), 'im_' + var ] = 1                                 
                    media = dataframe[var].mean()
                    dataframe[var] = dataframe[var].fillna( media )                          
            else:
                if var in vb.categ_vars:
                    if dummy:
                        dataframe[ 'im_' + var ] = 0
                        dataframe.loc[ dataframe[var].isnull(), 'im_' + var ] = 1  
                    moda = dataframe[var].mode()[ 0 ]
                    dataframe[var] = dataframe[var].fillna( moda )      
    return dataframe  



# 4. TRANSFORMACIONES
#####################
    
    
def imputar_outliers( dataframe, vars, percentil_superior ):
    '''
    Propósito:
        - Imputar los valores del "percentil_superior" asignado
          con 1 - "percentil_superior". Ejemplo: imputar los 
          valores del percentil superior 1% con el percentil 99%. 
    Inputs:
        - dataframes: lista de bases de datos
        - vars: variables que se busca imputar
        - percentil_superior: percentil asignado. Los valores que 
          sobrepasen este percentil serán imputados con 1 - "percentil_superior".
    Output:
        - Lista de bases de datos con aquellos valores del "percentil_superior" 
          asignado imputados con 1 - "percentil_superior". 
    Créditos:
        - Basado en la función "outliers_imputation" creada por el 
          profesor A. Cozzubo en: https://github.com/acozzubo/ML_Public_Policy
    '''
    for var in vars:
        if var in dataframe.columns:
            perc = dataframe[var].quantile( 1 - percentil_superior )
            dataframe.loc[ dataframe[var] > perc, var ] = perc
    return dataframe



def transformacion_log( datasets ):
    '''
    Propósito:
        - Realizar una transformación logarítmica a las variables
          provenientes de SIAF y a las variables dependientes numéricas.
    Inputs:
        - datasets: lista de bases de datos
    Output:
        - Lista de bases de datos con valores pertenecientes a variables
          de SIAF transformadas logarítmicamente.
    Especificaciones:
        - Para evitar que los valores transformados logarítmicamente tomen
          valores negativos, se suma 1 a todos los valores de las variables SIAF.
    '''
    for i, dataset in enumerate( datasets ):
        num_deps = [ "monto" ]
        log_cols = [ col for col in datasets[i].columns if col in vb.num_vars ]
        for var in log_cols:
            datasets[i][var] = np.log( datasets[i][var] + 1 )
    return datasets