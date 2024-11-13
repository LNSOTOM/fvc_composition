# -*- coding: utf-8 -*-
"""
***************************************************************************
    Grid.py
    ---------------------
    Date                 : May 2010
    Copyright            : (C) 2010 by Michael Minn
    Email                : pyqgis at michaelminn dot com
***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

__author__ = 'Michael Minn'
__date__ = 'May 2010'
__copyright__ = '(C) 2010, Michael Minn'

'''modified version to align grid with code: @author: lauransotomayor'''

import os
import math

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import QVariant
from qgis.core import (QgsApplication,
                       QgsField,
                       QgsFeatureSink,
                       QgsFeature,
                       QgsGeometry,
                       QgsLineString,
                       QgsPoint,
                       QgsPointXY,
                       QgsWkbTypes,
                       QgsProcessing,
                       QgsProcessingException,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterExtent,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterDistance,
                       QgsProcessingParameterCrs,
                       QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterRasterLayer,
                       QgsFields)

from processing.algs.qgis.QgisAlgorithm import QgisAlgorithm


pluginPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]


class Grid(QgisAlgorithm):
    RASTER = 'RASTER'
    EXTENT = 'EXTENT'
    HSPACING = 'HSPACING'
    VSPACING = 'VSPACING'
    PIXWIDTH = 'PIXWIDTH'
    PIXHEIGHT = 'PIXHEIGHT'
    HOVERLAY = 'HOVERLAY'
    VOVERLAY = 'VOVERLAY'
    CRS = 'CRS'
    OUTPUT = 'OUTPUT'
    OUTPUT2 = 'OUTPUT2'

    def icon(self):
        return QgsApplication.getThemeIcon("/algorithms/mAlgorithmCreateGrid.svg")

    def svgIconPath(self):
        return QgsApplication.iconPath("/algorithms/mAlgorithmCreateGrid.svg")

    def tags(self):
        return self.tr('grid,lines,polygons,vector,create,fishnet,diamond,hexagon').split(',')

    def group(self):
        return self.tr('Gridding - chunk_tiling')

    def groupId(self):
        return 'vectorcreation_row'

    def __init__(self):
        super().__init__()

    def initAlgorithm(self, config=None):

        self.addParameter(QgsProcessingParameterRasterLayer("Raster"))

        self.addParameter(QgsProcessingParameterDistance(self.HSPACING,
                                                         self.tr('Horizontal spacing'),
                                                         0.0, self.CRS, False, 0, 1000000000.0))
        self.addParameter(QgsProcessingParameterDistance(self.VSPACING,
                                                         self.tr('Vertical spacing'),
                                                         0.0, self.CRS, False, 0, 1000000000.0))
        self.addParameter(QgsProcessingParameterDistance(self.PIXWIDTH,
                                                         self.tr('Horizontal pixels'),
                                                         600.0, self.CRS, False, 0, 1000000000.0))
        self.addParameter(QgsProcessingParameterDistance(self.PIXHEIGHT,
                                                         self.tr('Vertical pixels'),
                                                        600.0, self.CRS, False, 0, 1000000000.0))
        self.addParameter(QgsProcessingParameterDistance(self.HOVERLAY,
                                                         self.tr('Horizontal overlap'),
                                                         0.0, self.CRS, False, 0, 1000000000.0))
        self.addParameter(QgsProcessingParameterDistance(self.VOVERLAY,
                                                         self.tr('Vertical overlap'),
                                                         0.0, self.CRS, False, 0, 1000000000.0))

        self.addParameter(QgsProcessingParameterExtent(self.EXTENT, self.tr('Grid extent')))

        self.addParameter(QgsProcessingParameterCrs(self.CRS, 'Grid CRS', 'ProjectCrs'))

        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, self.tr('Grid'), type=QgsProcessing.TypeVectorPolygon))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT2, self.tr('Annotations'), type=QgsProcessing.TypeVectorPolygon))

    def name(self):
        return 'creategrid_row'

    def displayName(self):
        return self.tr('Create reference training data (by row)')

    def processAlgorithm(self, parameters, context, feedback):

        rasterLyr = self.parameterAsRasterLayer(parameters,self.RASTER,context)
        pixWidth = self.parameterAsDouble(parameters, self.PIXWIDTH, context)
        pixHeight = self.parameterAsDouble(parameters, self.PIXHEIGHT, context)
        hSpacing = self.parameterAsDouble(parameters, self.HSPACING, context)
        vSpacing = self.parameterAsDouble(parameters, self.VSPACING, context)
        hOverlay = self.parameterAsDouble(parameters, self.HOVERLAY, context)
        vOverlay = self.parameterAsDouble(parameters, self.VOVERLAY, context)
        
        gsd_x = rasterLyr.rasterUnitsPerPixelX()
        gsd_y = rasterLyr.rasterUnitsPerPixelY()
        
        crs = self.parameterAsCrs(parameters, self.CRS, context)
        bbox = self.parameterAsExtent(parameters, self.EXTENT, context, crs)

        if (hSpacing > 0 or vSpacing > 0):
            pass
        elif pixWidth >0 and pixHeight>0:
            hSpacing = pixWidth*gsd_x
            vSpacing = pixHeight*gsd_y
        else:
            raise QgsProcessingException(
                self.tr('Invalid grid spacing: {0}/{1}').format(hSpacing, vSpacing))

        if bbox.width() < hSpacing:
            raise QgsProcessingException(
                self.tr('Horizontal spacing is too large for the covered area'))

        if hSpacing <= hOverlay or vSpacing <= vOverlay:
            raise QgsProcessingException(
                self.tr('Invalid overlay: {0}/{1}').format(hOverlay, vOverlay))

        if bbox.height() < vSpacing:
            raise QgsProcessingException(
                self.tr('Vertical spacing is too large for the covered area'))

        grid_fields = QgsFields()
        grid_fields.append(QgsField('id', QVariant.Int, '', 10, 0))
        grid_fields.append(QgsField('annotated',QVariant.Int))
        
        anot_fields = QgsFields()
        anot_fields.append(QgsField('id', QVariant.Int, '', 10, 0))
        anot_fields.append(QgsField('class',QVariant.Int, '', 10, 0))
        #anot_fields.append(QgsField('class',QVariant.String, '', 10, 0))

        outputWkb = QgsWkbTypes.Polygon
        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context, grid_fields, outputWkb, crs)
        if sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT))
            
        (sink2, dest_id2) = self.parameterAsSink(parameters, self.OUTPUT2, context, anot_fields, outputWkb, crs)
        if sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT2))

        self._rectangleGrid(sink, bbox, hSpacing, vSpacing, hOverlay, vOverlay, feedback)

        return {self.OUTPUT: dest_id,self.OUTPUT2: dest_id2}
    

    def _rectangleGrid(self, sink, bbox, hSpacing, vSpacing, hOverlay, vOverlay, feedback):
        feat = QgsFeature()

        # Calculate the number of columns and rows for the grid
        columns = int(math.ceil(bbox.width() / (hSpacing - hOverlay)))
        rows = int(math.ceil(bbox.height() / (vSpacing - vOverlay)))

        cells = rows * columns
        count_update = cells * 0.05

        id = 0
        count = 0

        for row in range(rows):  # Iterate through each row from top to bottom
            if feedback.isCanceled():
                break

            y1 = bbox.yMaximum() - (row * vSpacing - row * vOverlay)
            y2 = y1 - vSpacing

            for col in range(columns):  # Iterate through each column from left to right within the current row
                x1 = bbox.xMinimum() + (col * hSpacing - col * hOverlay)
                x2 = x1 + hSpacing

                # Define the corners of the grid cell
                polyline = [QgsPointXY(x1, y1), QgsPointXY(x2, y1), QgsPointXY(x2, y2), QgsPointXY(x1, y2), QgsPointXY(x1, y1)]

                # Create a polygon from the corners and set it as the geometry of the feature
                feat.setGeometry(QgsGeometry.fromPolygonXY([polyline]))
                feat.setAttributes([id, 0])  # Assuming '0' signifies not annotated; adjust as necessary

                # Add the feature to the sink
                sink.addFeature(feat, QgsFeatureSink.FastInsert)

                id += 1
                count += 1
                if count % count_update == 0:
                    feedback.setProgress(int(count / cells * 100))


   