/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#ifndef XQWT_MULTI_DIAL_H
#define XQWT_MULTI_DIAL_H

#include <QLabel>

#include "qwt/qwt_global.h"
#include "qwt/qwt_dial.h"
#include "qwt/qwt_dial_needle.h"


class QWT_EXPORT XqwtMultiDial: public QwtDial
{
    Q_OBJECT

protected:
   
   QString m_unitsText;
   
   //QLabel * m_unitsTextLabel {nullptr};
   
public:
   explicit XqwtMultiDial( QWidget* parent = NULL );
   
   virtual ~XqwtMultiDial();

   void setNumNeedles( int numNeedles );

   void setNeedle( int needleNo, QwtDialNeedle * );

    //const QwtDialNeedle *needle( int needleNo ) const;
    //QwtDialNeedle *needle( int needleNo );

   QString unitsText();
   void unitsText( QString & utext);
   
public Q_SLOTS:
    void setValue( int needleNo,
                   double val
                 );

protected:
    virtual void drawNeedle( QPainter *, const QPointF &,
        double radius, double direction, QPalette::ColorGroup ) const;

    virtual void drawNeedle( QPainter *, int needleNo, const QPointF &,
        double radius, double direction, QPalette::ColorGroup ) const;

private:
    // use setHand instead
    void setNeedle( QwtDialNeedle * );
    
    //void setNeedleNum( int needleNo, QwtDialNeedle *);

    int d_numNeedles;
    
    QwtDialNeedle **d_needles;
    
    std::vector<double> d_values;
};

#endif
