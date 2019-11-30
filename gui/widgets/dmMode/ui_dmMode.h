/********************************************************************************
** Form generated from reading UI file 'dmMode.ui'
**
** Created by: Qt User Interface Compiler version 5.9.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DMMODE_H
#define UI_DMMODE_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include "qwt_slider.h"

QT_BEGIN_NAMESPACE

class Ui_dmMode
{
public:
    QGridLayout *gridLayout;
    QLabel *title;
    QLabel *channel;
    QVBoxLayout *verticalLayout;
    QGridLayout *grid_0;
    QLabel *modeCurrent_0;
    QLineEdit *modeTarget_0;
    QLabel *modeName_0;
    QwtSlider *modeSlider_0;
    QHBoxLayout *horizontalLayout;
    QSpacerItem *horizontalSpacer;
    QPushButton *modeZero_0;
    QSpacerItem *horizontalSpacer_2;
    QGridLayout *grid_1;
    QLabel *modeCurrent_1;
    QLineEdit *modeTarget_1;
    QLabel *modeName_1;
    QwtSlider *modeSlider_1;
    QHBoxLayout *horizontalLayout_2;
    QSpacerItem *horizontalSpacer_3;
    QPushButton *modeZero_1;
    QSpacerItem *horizontalSpacer_4;
    QGridLayout *grid_2;
    QLabel *modeCurrent_2;
    QLineEdit *modeTarget_2;
    QLabel *modeName_2;
    QwtSlider *modeSlider_2;
    QHBoxLayout *horizontalLayout_3;
    QSpacerItem *horizontalSpacer_5;
    QPushButton *modeZero_2;
    QSpacerItem *horizontalSpacer_6;
    QGridLayout *grid_3;
    QLabel *modeCurrent_3;
    QLineEdit *modeTarget_3;
    QLabel *modeName_3;
    QwtSlider *modeSlider_3;
    QHBoxLayout *horizontalLayout_4;
    QSpacerItem *horizontalSpacer_7;
    QPushButton *modeZero_3;
    QSpacerItem *horizontalSpacer_8;
    QGridLayout *grid_4;
    QLabel *modeCurrent_4;
    QLineEdit *modeTarget_4;
    QLabel *modeName_4;
    QwtSlider *modeSlider_4;
    QHBoxLayout *horizontalLayout_5;
    QSpacerItem *horizontalSpacer_9;
    QPushButton *modeZero_4;
    QSpacerItem *horizontalSpacer_10;
    QGridLayout *grid_5;
    QLabel *modeCurrent_5;
    QLineEdit *modeTarget_5;
    QLabel *modeName_5;
    QwtSlider *modeSlider_5;
    QHBoxLayout *horizontalLayout_6;
    QSpacerItem *horizontalSpacer_11;
    QPushButton *modeZero_5;
    QSpacerItem *horizontalSpacer_12;
    QGridLayout *grid_6;
    QLabel *modeCurrent_6;
    QLineEdit *modeTarget_6;
    QLabel *modeName_6;
    QHBoxLayout *horizontalLayout_7;
    QSpacerItem *horizontalSpacer_13;
    QPushButton *modeZero_6;
    QSpacerItem *horizontalSpacer_14;
    QwtSlider *modeSlider_6;
    QGridLayout *grid_7;
    QLabel *modeCurrent_7;
    QLineEdit *modeTarget_7;
    QHBoxLayout *horizontalLayout_8;
    QSpacerItem *horizontalSpacer_15;
    QPushButton *modeZero_7;
    QSpacerItem *horizontalSpacer_16;
    QLabel *modeName_7;
    QwtSlider *modeSlider_7;
    QGridLayout *grid_8;
    QLabel *modeCurrent_8;
    QLineEdit *modeTarget_8;
    QLabel *modeName_8;
    QwtSlider *modeSlider_8;
    QHBoxLayout *horizontalLayout_9;
    QSpacerItem *horizontalSpacer_17;
    QPushButton *modeZero_8;
    QSpacerItem *horizontalSpacer_18;
    QGridLayout *grid_9;
    QLabel *modeCurrent_9;
    QLineEdit *modeTarget_9;
    QLabel *modeName_9;
    QwtSlider *modeSlider_9;
    QHBoxLayout *horizontalLayout_10;
    QSpacerItem *horizontalSpacer_19;
    QPushButton *modeZero_9;
    QSpacerItem *horizontalSpacer_20;
    QGridLayout *grid_10;
    QLabel *modeCurrent_10;
    QLineEdit *modeTarget_10;
    QLabel *modeName_10;
    QwtSlider *modeSlider_10;
    QHBoxLayout *horizontalLayout_11;
    QSpacerItem *horizontalSpacer_21;
    QPushButton *modeZero_10;
    QSpacerItem *horizontalSpacer_22;
    QHBoxLayout *horizontalLayout_12;
    QSpacerItem *horizontalSpacer_23;
    QPushButton *modeZero_all;
    QSpacerItem *horizontalSpacer_24;

    void setupUi(QDialog *dmMode)
    {
        if (dmMode->objectName().isEmpty())
            dmMode->setObjectName(QStringLiteral("dmMode"));
        dmMode->resize(801, 918);
        QPalette palette;
        QBrush brush(QColor(32, 31, 31, 255));
        brush.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::Base, brush);
        QBrush brush1(QColor(0, 0, 0, 255));
        brush1.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::Window, brush1);
        palette.setBrush(QPalette::Inactive, QPalette::Base, brush);
        palette.setBrush(QPalette::Inactive, QPalette::Window, brush1);
        palette.setBrush(QPalette::Disabled, QPalette::Base, brush1);
        palette.setBrush(QPalette::Disabled, QPalette::Window, brush1);
        dmMode->setPalette(palette);
        gridLayout = new QGridLayout(dmMode);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        title = new QLabel(dmMode);
        title->setObjectName(QStringLiteral("title"));
        QFont font;
        font.setPointSize(16);
        font.setBold(true);
        font.setWeight(75);
        title->setFont(font);
        title->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(title, 0, 0, 1, 1);

        channel = new QLabel(dmMode);
        channel->setObjectName(QStringLiteral("channel"));
        channel->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(channel, 1, 0, 1, 1);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        grid_0 = new QGridLayout();
        grid_0->setObjectName(QStringLiteral("grid_0"));
        grid_0->setSizeConstraint(QLayout::SetNoConstraint);
        modeCurrent_0 = new QLabel(dmMode);
        modeCurrent_0->setObjectName(QStringLiteral("modeCurrent_0"));
        modeCurrent_0->setFrameShape(QFrame::Panel);
        modeCurrent_0->setFrameShadow(QFrame::Sunken);
        modeCurrent_0->setTextFormat(Qt::PlainText);

        grid_0->addWidget(modeCurrent_0, 0, 3, 1, 1);

        modeTarget_0 = new QLineEdit(dmMode);
        modeTarget_0->setObjectName(QStringLiteral("modeTarget_0"));

        grid_0->addWidget(modeTarget_0, 0, 2, 1, 1);

        modeName_0 = new QLabel(dmMode);
        modeName_0->setObjectName(QStringLiteral("modeName_0"));

        grid_0->addWidget(modeName_0, 0, 0, 1, 1);

        modeSlider_0 = new QwtSlider(dmMode);
        modeSlider_0->setObjectName(QStringLiteral("modeSlider_0"));
        modeSlider_0->setLowerBound(-1);
        modeSlider_0->setUpperBound(1);
        modeSlider_0->setOrientation(Qt::Horizontal);
        modeSlider_0->setScalePosition(QwtSlider::NoScale);
        modeSlider_0->setTrough(false);
        modeSlider_0->setGroove(true);
        modeSlider_0->setHandleSize(QSize(15, 25));

        grid_0->addWidget(modeSlider_0, 0, 1, 1, 1);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        modeZero_0 = new QPushButton(dmMode);
        modeZero_0->setObjectName(QStringLiteral("modeZero_0"));
        QSizePolicy sizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(modeZero_0->sizePolicy().hasHeightForWidth());
        modeZero_0->setSizePolicy(sizePolicy);
        modeZero_0->setMinimumSize(QSize(0, 5));

        horizontalLayout->addWidget(modeZero_0);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_2);


        grid_0->addLayout(horizontalLayout, 1, 1, 1, 1);

        grid_0->setRowStretch(0, 3);
        grid_0->setRowStretch(1, 1);
        grid_0->setColumnStretch(0, 2);
        grid_0->setColumnStretch(1, 10);
        grid_0->setColumnStretch(2, 1);
        grid_0->setColumnStretch(3, 1);

        verticalLayout->addLayout(grid_0);

        grid_1 = new QGridLayout();
        grid_1->setObjectName(QStringLiteral("grid_1"));
        modeCurrent_1 = new QLabel(dmMode);
        modeCurrent_1->setObjectName(QStringLiteral("modeCurrent_1"));
        modeCurrent_1->setFrameShape(QFrame::Panel);
        modeCurrent_1->setFrameShadow(QFrame::Sunken);
        modeCurrent_1->setTextFormat(Qt::PlainText);

        grid_1->addWidget(modeCurrent_1, 0, 3, 1, 1);

        modeTarget_1 = new QLineEdit(dmMode);
        modeTarget_1->setObjectName(QStringLiteral("modeTarget_1"));

        grid_1->addWidget(modeTarget_1, 0, 2, 1, 1);

        modeName_1 = new QLabel(dmMode);
        modeName_1->setObjectName(QStringLiteral("modeName_1"));

        grid_1->addWidget(modeName_1, 0, 0, 1, 1);

        modeSlider_1 = new QwtSlider(dmMode);
        modeSlider_1->setObjectName(QStringLiteral("modeSlider_1"));
        modeSlider_1->setLowerBound(-1);
        modeSlider_1->setUpperBound(1);
        modeSlider_1->setOrientation(Qt::Horizontal);
        modeSlider_1->setScalePosition(QwtSlider::NoScale);
        modeSlider_1->setTrough(false);
        modeSlider_1->setGroove(true);
        modeSlider_1->setHandleSize(QSize(15, 25));

        grid_1->addWidget(modeSlider_1, 0, 1, 1, 1);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer_3);

        modeZero_1 = new QPushButton(dmMode);
        modeZero_1->setObjectName(QStringLiteral("modeZero_1"));
        sizePolicy.setHeightForWidth(modeZero_1->sizePolicy().hasHeightForWidth());
        modeZero_1->setSizePolicy(sizePolicy);
        modeZero_1->setMinimumSize(QSize(0, 5));

        horizontalLayout_2->addWidget(modeZero_1);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer_4);


        grid_1->addLayout(horizontalLayout_2, 1, 1, 1, 1);

        grid_1->setRowStretch(0, 3);
        grid_1->setRowStretch(1, 1);
        grid_1->setColumnStretch(0, 2);
        grid_1->setColumnStretch(1, 10);
        grid_1->setColumnStretch(2, 1);
        grid_1->setColumnStretch(3, 1);

        verticalLayout->addLayout(grid_1);

        grid_2 = new QGridLayout();
        grid_2->setObjectName(QStringLiteral("grid_2"));
        modeCurrent_2 = new QLabel(dmMode);
        modeCurrent_2->setObjectName(QStringLiteral("modeCurrent_2"));
        modeCurrent_2->setFrameShape(QFrame::Panel);
        modeCurrent_2->setFrameShadow(QFrame::Sunken);
        modeCurrent_2->setTextFormat(Qt::PlainText);

        grid_2->addWidget(modeCurrent_2, 0, 3, 1, 1);

        modeTarget_2 = new QLineEdit(dmMode);
        modeTarget_2->setObjectName(QStringLiteral("modeTarget_2"));

        grid_2->addWidget(modeTarget_2, 0, 2, 1, 1);

        modeName_2 = new QLabel(dmMode);
        modeName_2->setObjectName(QStringLiteral("modeName_2"));

        grid_2->addWidget(modeName_2, 0, 0, 1, 1);

        modeSlider_2 = new QwtSlider(dmMode);
        modeSlider_2->setObjectName(QStringLiteral("modeSlider_2"));
        modeSlider_2->setLowerBound(-1);
        modeSlider_2->setUpperBound(1);
        modeSlider_2->setOrientation(Qt::Horizontal);
        modeSlider_2->setScalePosition(QwtSlider::NoScale);
        modeSlider_2->setTrough(false);
        modeSlider_2->setGroove(true);
        modeSlider_2->setHandleSize(QSize(15, 25));

        grid_2->addWidget(modeSlider_2, 0, 1, 1, 1);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer_5);

        modeZero_2 = new QPushButton(dmMode);
        modeZero_2->setObjectName(QStringLiteral("modeZero_2"));
        sizePolicy.setHeightForWidth(modeZero_2->sizePolicy().hasHeightForWidth());
        modeZero_2->setSizePolicy(sizePolicy);
        modeZero_2->setMinimumSize(QSize(0, 5));

        horizontalLayout_3->addWidget(modeZero_2);

        horizontalSpacer_6 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer_6);


        grid_2->addLayout(horizontalLayout_3, 1, 1, 1, 1);

        grid_2->setRowStretch(0, 3);
        grid_2->setRowStretch(1, 1);
        grid_2->setColumnStretch(0, 2);
        grid_2->setColumnStretch(1, 10);
        grid_2->setColumnStretch(2, 1);
        grid_2->setColumnStretch(3, 1);

        verticalLayout->addLayout(grid_2);

        grid_3 = new QGridLayout();
        grid_3->setObjectName(QStringLiteral("grid_3"));
        modeCurrent_3 = new QLabel(dmMode);
        modeCurrent_3->setObjectName(QStringLiteral("modeCurrent_3"));
        modeCurrent_3->setFrameShape(QFrame::Panel);
        modeCurrent_3->setFrameShadow(QFrame::Sunken);
        modeCurrent_3->setTextFormat(Qt::PlainText);

        grid_3->addWidget(modeCurrent_3, 0, 3, 1, 1);

        modeTarget_3 = new QLineEdit(dmMode);
        modeTarget_3->setObjectName(QStringLiteral("modeTarget_3"));

        grid_3->addWidget(modeTarget_3, 0, 2, 1, 1);

        modeName_3 = new QLabel(dmMode);
        modeName_3->setObjectName(QStringLiteral("modeName_3"));

        grid_3->addWidget(modeName_3, 0, 0, 1, 1);

        modeSlider_3 = new QwtSlider(dmMode);
        modeSlider_3->setObjectName(QStringLiteral("modeSlider_3"));
        modeSlider_3->setLowerBound(-1);
        modeSlider_3->setUpperBound(1);
        modeSlider_3->setOrientation(Qt::Horizontal);
        modeSlider_3->setScalePosition(QwtSlider::NoScale);
        modeSlider_3->setTrough(false);
        modeSlider_3->setGroove(true);
        modeSlider_3->setHandleSize(QSize(15, 25));

        grid_3->addWidget(modeSlider_3, 0, 1, 1, 1);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QStringLiteral("horizontalLayout_4"));
        horizontalSpacer_7 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer_7);

        modeZero_3 = new QPushButton(dmMode);
        modeZero_3->setObjectName(QStringLiteral("modeZero_3"));
        sizePolicy.setHeightForWidth(modeZero_3->sizePolicy().hasHeightForWidth());
        modeZero_3->setSizePolicy(sizePolicy);
        modeZero_3->setMinimumSize(QSize(0, 5));

        horizontalLayout_4->addWidget(modeZero_3);

        horizontalSpacer_8 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer_8);


        grid_3->addLayout(horizontalLayout_4, 1, 1, 1, 1);

        grid_3->setRowStretch(0, 3);
        grid_3->setRowStretch(1, 1);
        grid_3->setColumnStretch(0, 2);
        grid_3->setColumnStretch(1, 10);
        grid_3->setColumnStretch(2, 1);
        grid_3->setColumnStretch(3, 1);

        verticalLayout->addLayout(grid_3);

        grid_4 = new QGridLayout();
        grid_4->setObjectName(QStringLiteral("grid_4"));
        modeCurrent_4 = new QLabel(dmMode);
        modeCurrent_4->setObjectName(QStringLiteral("modeCurrent_4"));
        modeCurrent_4->setFrameShape(QFrame::Panel);
        modeCurrent_4->setFrameShadow(QFrame::Sunken);
        modeCurrent_4->setTextFormat(Qt::PlainText);

        grid_4->addWidget(modeCurrent_4, 0, 3, 1, 1);

        modeTarget_4 = new QLineEdit(dmMode);
        modeTarget_4->setObjectName(QStringLiteral("modeTarget_4"));

        grid_4->addWidget(modeTarget_4, 0, 2, 1, 1);

        modeName_4 = new QLabel(dmMode);
        modeName_4->setObjectName(QStringLiteral("modeName_4"));

        grid_4->addWidget(modeName_4, 0, 0, 1, 1);

        modeSlider_4 = new QwtSlider(dmMode);
        modeSlider_4->setObjectName(QStringLiteral("modeSlider_4"));
        modeSlider_4->setLowerBound(-1);
        modeSlider_4->setUpperBound(1);
        modeSlider_4->setOrientation(Qt::Horizontal);
        modeSlider_4->setScalePosition(QwtSlider::NoScale);
        modeSlider_4->setTrough(false);
        modeSlider_4->setGroove(true);
        modeSlider_4->setHandleSize(QSize(15, 25));

        grid_4->addWidget(modeSlider_4, 0, 1, 1, 1);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName(QStringLiteral("horizontalLayout_5"));
        horizontalSpacer_9 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_5->addItem(horizontalSpacer_9);

        modeZero_4 = new QPushButton(dmMode);
        modeZero_4->setObjectName(QStringLiteral("modeZero_4"));
        sizePolicy.setHeightForWidth(modeZero_4->sizePolicy().hasHeightForWidth());
        modeZero_4->setSizePolicy(sizePolicy);
        modeZero_4->setMinimumSize(QSize(0, 5));

        horizontalLayout_5->addWidget(modeZero_4);

        horizontalSpacer_10 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_5->addItem(horizontalSpacer_10);


        grid_4->addLayout(horizontalLayout_5, 1, 1, 1, 1);

        grid_4->setRowStretch(0, 3);
        grid_4->setRowStretch(1, 1);
        grid_4->setColumnStretch(0, 2);
        grid_4->setColumnStretch(1, 10);
        grid_4->setColumnStretch(2, 1);
        grid_4->setColumnStretch(3, 1);

        verticalLayout->addLayout(grid_4);

        grid_5 = new QGridLayout();
        grid_5->setObjectName(QStringLiteral("grid_5"));
        modeCurrent_5 = new QLabel(dmMode);
        modeCurrent_5->setObjectName(QStringLiteral("modeCurrent_5"));
        modeCurrent_5->setFrameShape(QFrame::Panel);
        modeCurrent_5->setFrameShadow(QFrame::Sunken);
        modeCurrent_5->setTextFormat(Qt::PlainText);

        grid_5->addWidget(modeCurrent_5, 0, 3, 1, 1);

        modeTarget_5 = new QLineEdit(dmMode);
        modeTarget_5->setObjectName(QStringLiteral("modeTarget_5"));

        grid_5->addWidget(modeTarget_5, 0, 2, 1, 1);

        modeName_5 = new QLabel(dmMode);
        modeName_5->setObjectName(QStringLiteral("modeName_5"));

        grid_5->addWidget(modeName_5, 0, 0, 1, 1);

        modeSlider_5 = new QwtSlider(dmMode);
        modeSlider_5->setObjectName(QStringLiteral("modeSlider_5"));
        modeSlider_5->setLowerBound(-1);
        modeSlider_5->setUpperBound(1);
        modeSlider_5->setOrientation(Qt::Horizontal);
        modeSlider_5->setScalePosition(QwtSlider::NoScale);
        modeSlider_5->setTrough(false);
        modeSlider_5->setGroove(true);
        modeSlider_5->setHandleSize(QSize(15, 25));

        grid_5->addWidget(modeSlider_5, 0, 1, 1, 1);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setObjectName(QStringLiteral("horizontalLayout_6"));
        horizontalSpacer_11 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_6->addItem(horizontalSpacer_11);

        modeZero_5 = new QPushButton(dmMode);
        modeZero_5->setObjectName(QStringLiteral("modeZero_5"));
        sizePolicy.setHeightForWidth(modeZero_5->sizePolicy().hasHeightForWidth());
        modeZero_5->setSizePolicy(sizePolicy);
        modeZero_5->setMinimumSize(QSize(0, 5));

        horizontalLayout_6->addWidget(modeZero_5);

        horizontalSpacer_12 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_6->addItem(horizontalSpacer_12);


        grid_5->addLayout(horizontalLayout_6, 1, 1, 1, 1);

        grid_5->setRowStretch(0, 3);
        grid_5->setRowStretch(1, 1);
        grid_5->setColumnStretch(0, 2);
        grid_5->setColumnStretch(1, 10);
        grid_5->setColumnStretch(2, 1);
        grid_5->setColumnStretch(3, 1);

        verticalLayout->addLayout(grid_5);

        grid_6 = new QGridLayout();
        grid_6->setObjectName(QStringLiteral("grid_6"));
        modeCurrent_6 = new QLabel(dmMode);
        modeCurrent_6->setObjectName(QStringLiteral("modeCurrent_6"));
        modeCurrent_6->setFrameShape(QFrame::Panel);
        modeCurrent_6->setFrameShadow(QFrame::Sunken);
        modeCurrent_6->setTextFormat(Qt::PlainText);

        grid_6->addWidget(modeCurrent_6, 0, 3, 1, 1);

        modeTarget_6 = new QLineEdit(dmMode);
        modeTarget_6->setObjectName(QStringLiteral("modeTarget_6"));

        grid_6->addWidget(modeTarget_6, 0, 2, 1, 1);

        modeName_6 = new QLabel(dmMode);
        modeName_6->setObjectName(QStringLiteral("modeName_6"));

        grid_6->addWidget(modeName_6, 0, 0, 1, 1);

        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setObjectName(QStringLiteral("horizontalLayout_7"));
        horizontalSpacer_13 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_7->addItem(horizontalSpacer_13);

        modeZero_6 = new QPushButton(dmMode);
        modeZero_6->setObjectName(QStringLiteral("modeZero_6"));
        sizePolicy.setHeightForWidth(modeZero_6->sizePolicy().hasHeightForWidth());
        modeZero_6->setSizePolicy(sizePolicy);
        modeZero_6->setMinimumSize(QSize(0, 5));

        horizontalLayout_7->addWidget(modeZero_6);

        horizontalSpacer_14 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_7->addItem(horizontalSpacer_14);


        grid_6->addLayout(horizontalLayout_7, 1, 1, 1, 1);

        modeSlider_6 = new QwtSlider(dmMode);
        modeSlider_6->setObjectName(QStringLiteral("modeSlider_6"));
        modeSlider_6->setLowerBound(-1);
        modeSlider_6->setUpperBound(1);
        modeSlider_6->setOrientation(Qt::Horizontal);
        modeSlider_6->setScalePosition(QwtSlider::NoScale);
        modeSlider_6->setTrough(false);
        modeSlider_6->setGroove(true);
        modeSlider_6->setHandleSize(QSize(15, 25));

        grid_6->addWidget(modeSlider_6, 0, 1, 1, 1);

        grid_6->setRowStretch(0, 3);
        grid_6->setRowStretch(1, 1);
        grid_6->setColumnStretch(0, 2);
        grid_6->setColumnStretch(1, 10);
        grid_6->setColumnStretch(2, 1);
        grid_6->setColumnStretch(3, 1);

        verticalLayout->addLayout(grid_6);

        grid_7 = new QGridLayout();
        grid_7->setObjectName(QStringLiteral("grid_7"));
        modeCurrent_7 = new QLabel(dmMode);
        modeCurrent_7->setObjectName(QStringLiteral("modeCurrent_7"));
        modeCurrent_7->setFrameShape(QFrame::Panel);
        modeCurrent_7->setFrameShadow(QFrame::Sunken);
        modeCurrent_7->setTextFormat(Qt::PlainText);

        grid_7->addWidget(modeCurrent_7, 0, 3, 1, 1);

        modeTarget_7 = new QLineEdit(dmMode);
        modeTarget_7->setObjectName(QStringLiteral("modeTarget_7"));

        grid_7->addWidget(modeTarget_7, 0, 2, 1, 1);

        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setObjectName(QStringLiteral("horizontalLayout_8"));
        horizontalSpacer_15 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_8->addItem(horizontalSpacer_15);

        modeZero_7 = new QPushButton(dmMode);
        modeZero_7->setObjectName(QStringLiteral("modeZero_7"));
        sizePolicy.setHeightForWidth(modeZero_7->sizePolicy().hasHeightForWidth());
        modeZero_7->setSizePolicy(sizePolicy);
        modeZero_7->setMinimumSize(QSize(0, 5));

        horizontalLayout_8->addWidget(modeZero_7);

        horizontalSpacer_16 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_8->addItem(horizontalSpacer_16);


        grid_7->addLayout(horizontalLayout_8, 1, 1, 1, 1);

        modeName_7 = new QLabel(dmMode);
        modeName_7->setObjectName(QStringLiteral("modeName_7"));

        grid_7->addWidget(modeName_7, 0, 0, 1, 1);

        modeSlider_7 = new QwtSlider(dmMode);
        modeSlider_7->setObjectName(QStringLiteral("modeSlider_7"));
        modeSlider_7->setLowerBound(-1);
        modeSlider_7->setUpperBound(1);
        modeSlider_7->setOrientation(Qt::Horizontal);
        modeSlider_7->setScalePosition(QwtSlider::NoScale);
        modeSlider_7->setTrough(false);
        modeSlider_7->setGroove(true);
        modeSlider_7->setHandleSize(QSize(15, 25));

        grid_7->addWidget(modeSlider_7, 0, 1, 1, 1);

        grid_7->setRowStretch(0, 3);
        grid_7->setColumnStretch(0, 2);
        grid_7->setColumnStretch(1, 10);
        grid_7->setColumnStretch(2, 1);
        grid_7->setColumnStretch(3, 1);

        verticalLayout->addLayout(grid_7);

        grid_8 = new QGridLayout();
        grid_8->setObjectName(QStringLiteral("grid_8"));
        modeCurrent_8 = new QLabel(dmMode);
        modeCurrent_8->setObjectName(QStringLiteral("modeCurrent_8"));
        modeCurrent_8->setFrameShape(QFrame::Panel);
        modeCurrent_8->setFrameShadow(QFrame::Sunken);
        modeCurrent_8->setTextFormat(Qt::PlainText);

        grid_8->addWidget(modeCurrent_8, 0, 3, 1, 1);

        modeTarget_8 = new QLineEdit(dmMode);
        modeTarget_8->setObjectName(QStringLiteral("modeTarget_8"));

        grid_8->addWidget(modeTarget_8, 0, 2, 1, 1);

        modeName_8 = new QLabel(dmMode);
        modeName_8->setObjectName(QStringLiteral("modeName_8"));

        grid_8->addWidget(modeName_8, 0, 0, 1, 1);

        modeSlider_8 = new QwtSlider(dmMode);
        modeSlider_8->setObjectName(QStringLiteral("modeSlider_8"));
        modeSlider_8->setLowerBound(-1);
        modeSlider_8->setUpperBound(1);
        modeSlider_8->setOrientation(Qt::Horizontal);
        modeSlider_8->setScalePosition(QwtSlider::NoScale);
        modeSlider_8->setTrough(false);
        modeSlider_8->setGroove(true);
        modeSlider_8->setHandleSize(QSize(15, 25));

        grid_8->addWidget(modeSlider_8, 0, 1, 1, 1);

        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setObjectName(QStringLiteral("horizontalLayout_9"));
        horizontalSpacer_17 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_9->addItem(horizontalSpacer_17);

        modeZero_8 = new QPushButton(dmMode);
        modeZero_8->setObjectName(QStringLiteral("modeZero_8"));
        sizePolicy.setHeightForWidth(modeZero_8->sizePolicy().hasHeightForWidth());
        modeZero_8->setSizePolicy(sizePolicy);
        modeZero_8->setMinimumSize(QSize(0, 5));

        horizontalLayout_9->addWidget(modeZero_8);

        horizontalSpacer_18 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_9->addItem(horizontalSpacer_18);


        grid_8->addLayout(horizontalLayout_9, 1, 1, 1, 1);

        grid_8->setRowStretch(0, 3);
        grid_8->setRowStretch(1, 1);
        grid_8->setColumnStretch(0, 2);
        grid_8->setColumnStretch(1, 10);
        grid_8->setColumnStretch(2, 1);
        grid_8->setColumnStretch(3, 1);

        verticalLayout->addLayout(grid_8);

        grid_9 = new QGridLayout();
        grid_9->setObjectName(QStringLiteral("grid_9"));
        modeCurrent_9 = new QLabel(dmMode);
        modeCurrent_9->setObjectName(QStringLiteral("modeCurrent_9"));
        modeCurrent_9->setFrameShape(QFrame::Panel);
        modeCurrent_9->setFrameShadow(QFrame::Sunken);
        modeCurrent_9->setTextFormat(Qt::PlainText);

        grid_9->addWidget(modeCurrent_9, 0, 3, 1, 1);

        modeTarget_9 = new QLineEdit(dmMode);
        modeTarget_9->setObjectName(QStringLiteral("modeTarget_9"));

        grid_9->addWidget(modeTarget_9, 0, 2, 1, 1);

        modeName_9 = new QLabel(dmMode);
        modeName_9->setObjectName(QStringLiteral("modeName_9"));

        grid_9->addWidget(modeName_9, 0, 0, 1, 1);

        modeSlider_9 = new QwtSlider(dmMode);
        modeSlider_9->setObjectName(QStringLiteral("modeSlider_9"));
        modeSlider_9->setLowerBound(-1);
        modeSlider_9->setUpperBound(1);
        modeSlider_9->setOrientation(Qt::Horizontal);
        modeSlider_9->setScalePosition(QwtSlider::NoScale);
        modeSlider_9->setTrough(false);
        modeSlider_9->setGroove(true);
        modeSlider_9->setHandleSize(QSize(15, 25));

        grid_9->addWidget(modeSlider_9, 0, 1, 1, 1);

        horizontalLayout_10 = new QHBoxLayout();
        horizontalLayout_10->setObjectName(QStringLiteral("horizontalLayout_10"));
        horizontalSpacer_19 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_10->addItem(horizontalSpacer_19);

        modeZero_9 = new QPushButton(dmMode);
        modeZero_9->setObjectName(QStringLiteral("modeZero_9"));
        sizePolicy.setHeightForWidth(modeZero_9->sizePolicy().hasHeightForWidth());
        modeZero_9->setSizePolicy(sizePolicy);
        modeZero_9->setMinimumSize(QSize(0, 5));

        horizontalLayout_10->addWidget(modeZero_9);

        horizontalSpacer_20 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_10->addItem(horizontalSpacer_20);


        grid_9->addLayout(horizontalLayout_10, 1, 1, 1, 1);

        grid_9->setRowStretch(0, 3);
        grid_9->setRowStretch(1, 1);
        grid_9->setColumnStretch(0, 2);
        grid_9->setColumnStretch(1, 10);
        grid_9->setColumnStretch(2, 1);
        grid_9->setColumnStretch(3, 1);

        verticalLayout->addLayout(grid_9);

        grid_10 = new QGridLayout();
        grid_10->setObjectName(QStringLiteral("grid_10"));
        modeCurrent_10 = new QLabel(dmMode);
        modeCurrent_10->setObjectName(QStringLiteral("modeCurrent_10"));
        modeCurrent_10->setFrameShape(QFrame::Panel);
        modeCurrent_10->setFrameShadow(QFrame::Sunken);
        modeCurrent_10->setTextFormat(Qt::PlainText);

        grid_10->addWidget(modeCurrent_10, 0, 3, 1, 1);

        modeTarget_10 = new QLineEdit(dmMode);
        modeTarget_10->setObjectName(QStringLiteral("modeTarget_10"));

        grid_10->addWidget(modeTarget_10, 0, 2, 1, 1);

        modeName_10 = new QLabel(dmMode);
        modeName_10->setObjectName(QStringLiteral("modeName_10"));

        grid_10->addWidget(modeName_10, 0, 0, 1, 1);

        modeSlider_10 = new QwtSlider(dmMode);
        modeSlider_10->setObjectName(QStringLiteral("modeSlider_10"));
        modeSlider_10->setLowerBound(-1);
        modeSlider_10->setUpperBound(1);
        modeSlider_10->setOrientation(Qt::Horizontal);
        modeSlider_10->setScalePosition(QwtSlider::NoScale);
        modeSlider_10->setTrough(false);
        modeSlider_10->setGroove(true);
        modeSlider_10->setHandleSize(QSize(15, 25));

        grid_10->addWidget(modeSlider_10, 0, 1, 1, 1);

        horizontalLayout_11 = new QHBoxLayout();
        horizontalLayout_11->setObjectName(QStringLiteral("horizontalLayout_11"));
        horizontalSpacer_21 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_11->addItem(horizontalSpacer_21);

        modeZero_10 = new QPushButton(dmMode);
        modeZero_10->setObjectName(QStringLiteral("modeZero_10"));
        sizePolicy.setHeightForWidth(modeZero_10->sizePolicy().hasHeightForWidth());
        modeZero_10->setSizePolicy(sizePolicy);
        modeZero_10->setMinimumSize(QSize(0, 5));

        horizontalLayout_11->addWidget(modeZero_10);

        horizontalSpacer_22 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_11->addItem(horizontalSpacer_22);


        grid_10->addLayout(horizontalLayout_11, 1, 1, 1, 1);

        grid_10->setRowStretch(0, 3);
        grid_10->setRowStretch(1, 1);
        grid_10->setColumnStretch(0, 2);
        grid_10->setColumnStretch(1, 10);
        grid_10->setColumnStretch(2, 1);
        grid_10->setColumnStretch(3, 1);

        verticalLayout->addLayout(grid_10);


        gridLayout->addLayout(verticalLayout, 2, 0, 1, 1);

        horizontalLayout_12 = new QHBoxLayout();
        horizontalLayout_12->setObjectName(QStringLiteral("horizontalLayout_12"));
        horizontalSpacer_23 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_12->addItem(horizontalSpacer_23);

        modeZero_all = new QPushButton(dmMode);
        modeZero_all->setObjectName(QStringLiteral("modeZero_all"));
        sizePolicy.setHeightForWidth(modeZero_all->sizePolicy().hasHeightForWidth());
        modeZero_all->setSizePolicy(sizePolicy);
        modeZero_all->setMinimumSize(QSize(0, 5));

        horizontalLayout_12->addWidget(modeZero_all);

        horizontalSpacer_24 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_12->addItem(horizontalSpacer_24);


        gridLayout->addLayout(horizontalLayout_12, 3, 0, 1, 1);


        retranslateUi(dmMode);

        QMetaObject::connectSlotsByName(dmMode);
    } // setupUi

    void retranslateUi(QDialog *dmMode)
    {
        dmMode->setWindowTitle(QApplication::translate("dmMode", "Dialog", Q_NULLPTR));
        title->setText(QApplication::translate("dmMode", "DM Modes", Q_NULLPTR));
        channel->setText(QApplication::translate("dmMode", "dmXXdispYY", Q_NULLPTR));
        modeCurrent_0->setText(QApplication::translate("dmMode", "0.0", Q_NULLPTR));
        modeName_0->setText(QApplication::translate("dmMode", "Mode 00", Q_NULLPTR));
        modeZero_0->setText(QApplication::translate("dmMode", "0", Q_NULLPTR));
        modeCurrent_1->setText(QApplication::translate("dmMode", "0.0", Q_NULLPTR));
        modeName_1->setText(QApplication::translate("dmMode", "Mode 01", Q_NULLPTR));
        modeZero_1->setText(QApplication::translate("dmMode", "0", Q_NULLPTR));
        modeCurrent_2->setText(QApplication::translate("dmMode", "0.0", Q_NULLPTR));
        modeName_2->setText(QApplication::translate("dmMode", "Mode 02", Q_NULLPTR));
        modeZero_2->setText(QApplication::translate("dmMode", "0", Q_NULLPTR));
        modeCurrent_3->setText(QApplication::translate("dmMode", "0.0", Q_NULLPTR));
        modeName_3->setText(QApplication::translate("dmMode", "Mode 03", Q_NULLPTR));
        modeZero_3->setText(QApplication::translate("dmMode", "0", Q_NULLPTR));
        modeCurrent_4->setText(QApplication::translate("dmMode", "0.0", Q_NULLPTR));
        modeName_4->setText(QApplication::translate("dmMode", "Mode 04", Q_NULLPTR));
        modeZero_4->setText(QApplication::translate("dmMode", "0", Q_NULLPTR));
        modeCurrent_5->setText(QApplication::translate("dmMode", "0.0", Q_NULLPTR));
        modeName_5->setText(QApplication::translate("dmMode", "Mode 05", Q_NULLPTR));
        modeZero_5->setText(QApplication::translate("dmMode", "0", Q_NULLPTR));
        modeCurrent_6->setText(QApplication::translate("dmMode", "0.0", Q_NULLPTR));
        modeName_6->setText(QApplication::translate("dmMode", "Mode 06", Q_NULLPTR));
        modeZero_6->setText(QApplication::translate("dmMode", "0", Q_NULLPTR));
        modeCurrent_7->setText(QApplication::translate("dmMode", "0.0", Q_NULLPTR));
        modeZero_7->setText(QApplication::translate("dmMode", "0", Q_NULLPTR));
        modeName_7->setText(QApplication::translate("dmMode", "Mode 07", Q_NULLPTR));
        modeCurrent_8->setText(QApplication::translate("dmMode", "0.0", Q_NULLPTR));
        modeName_8->setText(QApplication::translate("dmMode", "Mode 08", Q_NULLPTR));
        modeZero_8->setText(QApplication::translate("dmMode", "0", Q_NULLPTR));
        modeCurrent_9->setText(QApplication::translate("dmMode", "0.0", Q_NULLPTR));
        modeName_9->setText(QApplication::translate("dmMode", "Mode 09", Q_NULLPTR));
        modeZero_9->setText(QApplication::translate("dmMode", "0", Q_NULLPTR));
        modeCurrent_10->setText(QApplication::translate("dmMode", "0.0", Q_NULLPTR));
        modeName_10->setText(QApplication::translate("dmMode", "Mode 10", Q_NULLPTR));
        modeZero_10->setText(QApplication::translate("dmMode", "0", Q_NULLPTR));
        modeZero_all->setText(QApplication::translate("dmMode", "zero all", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class dmMode: public Ui_dmMode {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DMMODE_H
