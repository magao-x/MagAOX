/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "xqwt_multi_dial.h"
#include <qmath.h>
#include <qlocale.h>

#include <iostream>
/*!
  Constructor
  \param parent Parent widget
*/
XqwtMultiDial::XqwtMultiDial( QWidget *parent ): QwtDial( parent )
{
   setWrapping( true );
   setReadOnly( true );

   
   d_numNeedles = 0;
   d_needles = 0;
   
//    m_unitsTextLabel = new QLabel(this);
//    auto c = QWidget::rect().center();
//    std::cerr << "c: " << c.x() << " " << c.y() << "\n";
//    m_unitsTextLabel->setGeometry(c.x(),c.y()+10,80,30);
//    m_unitsTextLabel->setText("Amps");
//    m_unitsTextLabel->raise();
//    m_unitsTextLabel->show();

}

//! Destructor
XqwtMultiDial::~XqwtMultiDial()
{
    for ( int i = 0; i < d_numNeedles; i++ )
        delete d_needles[i];
    
    delete d_needles;
    
    //delete m_unitsTextLabel;
}

void XqwtMultiDial::setNumNeedles(int numNeedles)
{
   if(d_needles)
   {
      for(int i=0; i< d_numNeedles; ++i) delete d_needles[i];
      
      delete[] d_needles;
   }
   
   d_numNeedles = numNeedles;
   d_needles = new QwtDialNeedle*[d_numNeedles];
   
   QColor needleColor = palette().color( QPalette::Active, QPalette::BrightText );
   QColor knobColor = palette().color( QPalette::Active, QPalette::Text );
   //needleColor = Qt::red; //needleColor.dark( 120 );
   
   for ( int i = 0; i < d_numNeedles; i++ )
   {
      d_needles[i] = new QwtDialSimpleNeedle( QwtDialSimpleNeedle::Arrow, true, needleColor, knobColor );
   }
   
   d_values.resize( d_numNeedles, 0.0);
}


/*!
  Nop method, use setHand() instead
  \sa setHand()
*/
void XqwtMultiDial::setNeedle( QwtDialNeedle * )
{
    // no op
    return;
}

/*!
   Set a clock hand
   \param hand Specifies the type of hand
   \param needle Hand
   \sa hand()
*/
void XqwtMultiDial::setNeedle( int needleNo, 
                                  QwtDialNeedle *needle )
{
   if(d_needles == nullptr) return;
   
    if ( needleNo >= 0 && needleNo < d_numNeedles && needle)
    {
        delete d_needles[needleNo];
        d_needles[needleNo] = needle;
    }
}

void XqwtMultiDial::setValue( int needleNo,
                              double value
                            )
{
   if(needleNo < 0 || needleNo >= d_numNeedles) return;
   
   d_values[needleNo] = value;
   
   update();

}

/*!
  \brief Draw the needle

  A multiDial has no single needle but three hands instead. drawNeedle()
  translates value() into directions for the hands and calls
  drawHand().

  \param painter Painter
  \param center Center of the clock
  \param radius Maximum length for the hands
  \param dir Dummy, not used.
  \param colorGroup ColorGroup

  \sa drawHand()
*/
void XqwtMultiDial::drawNeedle( QPainter *painter, const QPointF &center,
    double radius, double dir, QPalette::ColorGroup colorGroup ) const
{
    Q_UNUSED( dir );

    if(d_needles == nullptr) return;
    
    if ( isValid() )
    {
        for ( int i = 0; i < d_numNeedles; ++i )
        {
           //std::cerr << d_values[i] << " " << maxScaleArc() << " " << minScaleArc() << " " << upperBound() << " " << lowerBound() << "\n";
            drawNeedle( painter, i,
                center, radius, 360.0 - ((d_values[i]-lowerBound())*(maxScaleArc()-minScaleArc())/(upperBound()-lowerBound())) - origin(), colorGroup );
        }
    }
}

/*!
  Draw a clock hand

  \param painter Painter
  \param hd Specify the type of hand
  \param center Center of the clock
  \param radius Maximum length for the hands
  \param direction Direction of the hand in degrees, counter clockwise
  \param cg ColorGroup
*/
void XqwtMultiDial::drawNeedle( QPainter *painter, int needlNo,
    const QPointF &center, double radius, double direction,
    QPalette::ColorGroup cg ) const
{
   if(d_needles == nullptr) return;
   
    const QwtDialNeedle *needle = d_needles[ needlNo ];
    if ( needle )
    {
        needle->draw( painter, center, radius, direction, cg );
    }
}


