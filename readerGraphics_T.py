import numpy as np

def searchAlphawithCy(cy, graph):    
    cy1 = 0.22
    pxa1 = 0
    cy2 = 0.08
    pxa2 = 255 
    px_cy = (cy-cy1)*(pxa2-pxa1)/(cy2-cy1)+pxa1
    px_cy = round(px_cy)
    
    pxc = np.argmax(graph[px_cy])
    alpha1 = -6
    pxc1 = 0
    alpha2 = 2
    pxc2 = 255
    alpha = (pxc-pxc1)*(alpha2-alpha1)/(pxc2-pxc1)+alpha1
    return alpha

def searchEwithCy(cy, graph):    
    cy1 = 0.08
    pxa1 = 0
    cy2 = 0.22
    pxa2 = 255 
    px_cy = (cy-cy1)*(pxa2-pxa1)/(cy2-cy1)+pxa1
    px_cy = round(px_cy)
    
    pxc = np.argmax(graph[:,px_cy])
    E1 = 28
    pxc1 = 0
    E2 = 0
    pxc2 = 255
    E = (pxc-pxc1)*(E2-E1)/(pxc2-pxc1)+E1
    return E
