/********************************************************************************
** Form generated from reading UI file 'pwr.ui'
**
** Created by: Qt User Interface Compiler version 5.9.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_PWR_H
#define UI_PWR_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLCDNumber>
#include <QtWidgets/QLabel>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "xqwt_multi_dial.h"

QT_BEGIN_NAMESPACE

class Ui_pwr
{
public:
    QGridLayout *gridLayout_2;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    XqwtMultiDial *currentDial;
    XqwtMultiDial *voltageDial;
    XqwtMultiDial *frequencyDial;
    QHBoxLayout *layoutValues;
    QHBoxLayout *layoutAmps;
    QLabel *ampsLabel;
    QLCDNumber *totalCurrent;
    QHBoxLayout *layoutVolts;
    QLabel *ampsLabel_2;
    QLCDNumber *averageVoltage;
    QHBoxLayout *layoutHz;
    QLabel *ampsLabel_3;
    QLCDNumber *averageFrequency;
    QSpacerItem *verticalSpacer;
    QGridLayout *switchGrid;

    void setupUi(QWidget *pwr)
    {
        if (pwr->objectName().isEmpty())
            pwr->setObjectName(QStringLiteral("pwr"));
        pwr->resize(814, 637);
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
        pwr->setPalette(palette);
        gridLayout_2 = new QGridLayout(pwr);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setSizeConstraint(QLayout::SetDefaultConstraint);
        currentDial = new XqwtMultiDial(pwr);
        currentDial->setObjectName(QStringLiteral("currentDial"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(1);
        sizePolicy.setVerticalStretch(2);
        sizePolicy.setHeightForWidth(currentDial->sizePolicy().hasHeightForWidth());
        currentDial->setSizePolicy(sizePolicy);
        currentDial->setMinimumSize(QSize(250, 250));
        currentDial->setMaximumSize(QSize(250, 250));
        QPalette palette1;
        QBrush brush2(QColor(224, 222, 219, 255));
        brush2.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Active, QPalette::WindowText, brush2);
        QBrush brush3(QColor(243, 240, 236, 255));
        brush3.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Active, QPalette::Button, brush3);
        QBrush brush4(QColor(244, 238, 238, 255));
        brush4.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Active, QPalette::Light, brush4);
        QBrush brush5(QColor(229, 221, 221, 255));
        brush5.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Active, QPalette::Dark, brush5);
        QBrush brush6(QColor(9, 175, 175, 255));
        brush6.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Active, QPalette::Text, brush6);
        palette1.setBrush(QPalette::Active, QPalette::BrightText, brush6);
        QBrush brush7(QColor(232, 230, 227, 255));
        brush7.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Active, QPalette::ButtonText, brush7);
        QBrush brush8(QColor(253, 246, 246, 255));
        brush8.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Active, QPalette::Base, brush8);
        QBrush brush9(QColor(232, 225, 225, 255));
        brush9.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Active, QPalette::Window, brush9);
        QBrush brush10(QColor(240, 234, 234, 255));
        brush10.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Active, QPalette::AlternateBase, brush10);
        QBrush brush11(QColor(244, 244, 244, 255));
        brush11.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Active, QPalette::NoRole, brush11);
        palette1.setBrush(QPalette::Inactive, QPalette::WindowText, brush2);
        palette1.setBrush(QPalette::Inactive, QPalette::Button, brush3);
        palette1.setBrush(QPalette::Inactive, QPalette::Light, brush4);
        palette1.setBrush(QPalette::Inactive, QPalette::Dark, brush5);
        palette1.setBrush(QPalette::Inactive, QPalette::Text, brush6);
        palette1.setBrush(QPalette::Inactive, QPalette::BrightText, brush6);
        palette1.setBrush(QPalette::Inactive, QPalette::ButtonText, brush7);
        palette1.setBrush(QPalette::Inactive, QPalette::Base, brush8);
        palette1.setBrush(QPalette::Inactive, QPalette::Window, brush9);
        palette1.setBrush(QPalette::Inactive, QPalette::AlternateBase, brush10);
        palette1.setBrush(QPalette::Inactive, QPalette::NoRole, brush11);
        palette1.setBrush(QPalette::Disabled, QPalette::WindowText, brush5);
        palette1.setBrush(QPalette::Disabled, QPalette::Button, brush3);
        palette1.setBrush(QPalette::Disabled, QPalette::Light, brush4);
        palette1.setBrush(QPalette::Disabled, QPalette::Dark, brush5);
        palette1.setBrush(QPalette::Disabled, QPalette::Text, brush5);
        palette1.setBrush(QPalette::Disabled, QPalette::BrightText, brush6);
        palette1.setBrush(QPalette::Disabled, QPalette::ButtonText, brush5);
        palette1.setBrush(QPalette::Disabled, QPalette::Base, brush9);
        palette1.setBrush(QPalette::Disabled, QPalette::Window, brush9);
        palette1.setBrush(QPalette::Disabled, QPalette::AlternateBase, brush10);
        palette1.setBrush(QPalette::Disabled, QPalette::NoRole, brush11);
        currentDial->setPalette(palette1);

        horizontalLayout->addWidget(currentDial);

        voltageDial = new XqwtMultiDial(pwr);
        voltageDial->setObjectName(QStringLiteral("voltageDial"));
        sizePolicy.setHeightForWidth(voltageDial->sizePolicy().hasHeightForWidth());
        voltageDial->setSizePolicy(sizePolicy);
        voltageDial->setMinimumSize(QSize(250, 250));
        voltageDial->setMaximumSize(QSize(250, 250));
        QPalette palette2;
        palette2.setBrush(QPalette::Active, QPalette::WindowText, brush2);
        palette2.setBrush(QPalette::Active, QPalette::Button, brush3);
        palette2.setBrush(QPalette::Active, QPalette::Light, brush4);
        palette2.setBrush(QPalette::Active, QPalette::Dark, brush5);
        palette2.setBrush(QPalette::Active, QPalette::Text, brush6);
        palette2.setBrush(QPalette::Active, QPalette::BrightText, brush6);
        palette2.setBrush(QPalette::Active, QPalette::ButtonText, brush7);
        palette2.setBrush(QPalette::Active, QPalette::Base, brush8);
        palette2.setBrush(QPalette::Active, QPalette::Window, brush9);
        palette2.setBrush(QPalette::Active, QPalette::AlternateBase, brush10);
        palette2.setBrush(QPalette::Active, QPalette::NoRole, brush11);
        palette2.setBrush(QPalette::Inactive, QPalette::WindowText, brush2);
        palette2.setBrush(QPalette::Inactive, QPalette::Button, brush3);
        palette2.setBrush(QPalette::Inactive, QPalette::Light, brush4);
        palette2.setBrush(QPalette::Inactive, QPalette::Dark, brush5);
        palette2.setBrush(QPalette::Inactive, QPalette::Text, brush6);
        palette2.setBrush(QPalette::Inactive, QPalette::BrightText, brush6);
        palette2.setBrush(QPalette::Inactive, QPalette::ButtonText, brush7);
        palette2.setBrush(QPalette::Inactive, QPalette::Base, brush8);
        palette2.setBrush(QPalette::Inactive, QPalette::Window, brush9);
        palette2.setBrush(QPalette::Inactive, QPalette::AlternateBase, brush10);
        palette2.setBrush(QPalette::Inactive, QPalette::NoRole, brush11);
        palette2.setBrush(QPalette::Disabled, QPalette::WindowText, brush5);
        palette2.setBrush(QPalette::Disabled, QPalette::Button, brush3);
        palette2.setBrush(QPalette::Disabled, QPalette::Light, brush4);
        palette2.setBrush(QPalette::Disabled, QPalette::Dark, brush5);
        palette2.setBrush(QPalette::Disabled, QPalette::Text, brush5);
        palette2.setBrush(QPalette::Disabled, QPalette::BrightText, brush6);
        palette2.setBrush(QPalette::Disabled, QPalette::ButtonText, brush5);
        palette2.setBrush(QPalette::Disabled, QPalette::Base, brush9);
        palette2.setBrush(QPalette::Disabled, QPalette::Window, brush9);
        palette2.setBrush(QPalette::Disabled, QPalette::AlternateBase, brush10);
        palette2.setBrush(QPalette::Disabled, QPalette::NoRole, brush11);
        voltageDial->setPalette(palette2);

        horizontalLayout->addWidget(voltageDial);

        frequencyDial = new XqwtMultiDial(pwr);
        frequencyDial->setObjectName(QStringLiteral("frequencyDial"));
        sizePolicy.setHeightForWidth(frequencyDial->sizePolicy().hasHeightForWidth());
        frequencyDial->setSizePolicy(sizePolicy);
        frequencyDial->setMinimumSize(QSize(250, 250));
        frequencyDial->setMaximumSize(QSize(250, 250));
        QPalette palette3;
        palette3.setBrush(QPalette::Active, QPalette::WindowText, brush2);
        palette3.setBrush(QPalette::Active, QPalette::Button, brush3);
        palette3.setBrush(QPalette::Active, QPalette::Light, brush4);
        palette3.setBrush(QPalette::Active, QPalette::Dark, brush5);
        palette3.setBrush(QPalette::Active, QPalette::Text, brush6);
        palette3.setBrush(QPalette::Active, QPalette::BrightText, brush6);
        palette3.setBrush(QPalette::Active, QPalette::ButtonText, brush7);
        palette3.setBrush(QPalette::Active, QPalette::Base, brush8);
        palette3.setBrush(QPalette::Active, QPalette::Window, brush9);
        palette3.setBrush(QPalette::Active, QPalette::AlternateBase, brush10);
        palette3.setBrush(QPalette::Active, QPalette::NoRole, brush11);
        palette3.setBrush(QPalette::Inactive, QPalette::WindowText, brush2);
        palette3.setBrush(QPalette::Inactive, QPalette::Button, brush3);
        palette3.setBrush(QPalette::Inactive, QPalette::Light, brush4);
        palette3.setBrush(QPalette::Inactive, QPalette::Dark, brush5);
        palette3.setBrush(QPalette::Inactive, QPalette::Text, brush6);
        palette3.setBrush(QPalette::Inactive, QPalette::BrightText, brush6);
        palette3.setBrush(QPalette::Inactive, QPalette::ButtonText, brush7);
        palette3.setBrush(QPalette::Inactive, QPalette::Base, brush8);
        palette3.setBrush(QPalette::Inactive, QPalette::Window, brush9);
        palette3.setBrush(QPalette::Inactive, QPalette::AlternateBase, brush10);
        palette3.setBrush(QPalette::Inactive, QPalette::NoRole, brush11);
        palette3.setBrush(QPalette::Disabled, QPalette::WindowText, brush5);
        palette3.setBrush(QPalette::Disabled, QPalette::Button, brush3);
        palette3.setBrush(QPalette::Disabled, QPalette::Light, brush4);
        palette3.setBrush(QPalette::Disabled, QPalette::Dark, brush5);
        palette3.setBrush(QPalette::Disabled, QPalette::Text, brush5);
        palette3.setBrush(QPalette::Disabled, QPalette::BrightText, brush6);
        palette3.setBrush(QPalette::Disabled, QPalette::ButtonText, brush5);
        palette3.setBrush(QPalette::Disabled, QPalette::Base, brush9);
        palette3.setBrush(QPalette::Disabled, QPalette::Window, brush9);
        palette3.setBrush(QPalette::Disabled, QPalette::AlternateBase, brush10);
        palette3.setBrush(QPalette::Disabled, QPalette::NoRole, brush11);
        frequencyDial->setPalette(palette3);

        horizontalLayout->addWidget(frequencyDial);

        horizontalLayout->setStretch(0, 1);
        horizontalLayout->setStretch(1, 1);
        horizontalLayout->setStretch(2, 1);

        verticalLayout->addLayout(horizontalLayout);

        layoutValues = new QHBoxLayout();
        layoutValues->setObjectName(QStringLiteral("layoutValues"));
        layoutAmps = new QHBoxLayout();
        layoutAmps->setObjectName(QStringLiteral("layoutAmps"));
        ampsLabel = new QLabel(pwr);
        ampsLabel->setObjectName(QStringLiteral("ampsLabel"));
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::Maximum);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(ampsLabel->sizePolicy().hasHeightForWidth());
        ampsLabel->setSizePolicy(sizePolicy1);
        ampsLabel->setMaximumSize(QSize(100, 50));
        QPalette palette4;
        palette4.setBrush(QPalette::Active, QPalette::WindowText, brush6);
        palette4.setBrush(QPalette::Inactive, QPalette::WindowText, brush6);
        QBrush brush12(QColor(96, 95, 94, 255));
        brush12.setStyle(Qt::SolidPattern);
        palette4.setBrush(QPalette::Disabled, QPalette::WindowText, brush12);
        ampsLabel->setPalette(palette4);
        ampsLabel->setAlignment(Qt::AlignCenter);

        layoutAmps->addWidget(ampsLabel);

        totalCurrent = new QLCDNumber(pwr);
        totalCurrent->setObjectName(QStringLiteral("totalCurrent"));
        QSizePolicy sizePolicy2(QSizePolicy::Minimum, QSizePolicy::Maximum);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(totalCurrent->sizePolicy().hasHeightForWidth());
        totalCurrent->setSizePolicy(sizePolicy2);
        totalCurrent->setMinimumSize(QSize(0, 50));
        totalCurrent->setMaximumSize(QSize(100, 50));
        QPalette palette5;
        palette5.setBrush(QPalette::Active, QPalette::WindowText, brush6);
        QBrush brush13(QColor(64, 63, 62, 0));
        brush13.setStyle(Qt::SolidPattern);
        palette5.setBrush(QPalette::Active, QPalette::Button, brush13);
        QBrush brush14(QColor(4, 3, 3, 0));
        brush14.setStyle(Qt::SolidPattern);
        palette5.setBrush(QPalette::Active, QPalette::Light, brush14);
        QBrush brush15(QColor(61, 60, 60, 0));
        brush15.setStyle(Qt::SolidPattern);
        palette5.setBrush(QPalette::Active, QPalette::Midlight, brush15);
        QBrush brush16(QColor(28, 25, 16, 0));
        brush16.setStyle(Qt::SolidPattern);
        palette5.setBrush(QPalette::Active, QPalette::Dark, brush16);
        QBrush brush17(QColor(212, 210, 207, 255));
        brush17.setStyle(Qt::SolidPattern);
        palette5.setBrush(QPalette::Active, QPalette::Text, brush17);
        palette5.setBrush(QPalette::Active, QPalette::ButtonText, brush7);
        palette5.setBrush(QPalette::Active, QPalette::Base, brush);
        QBrush brush18(QColor(30, 29, 29, 0));
        brush18.setStyle(Qt::SolidPattern);
        palette5.setBrush(QPalette::Active, QPalette::Window, brush18);
        palette5.setBrush(QPalette::Inactive, QPalette::WindowText, brush6);
        palette5.setBrush(QPalette::Inactive, QPalette::Button, brush13);
        palette5.setBrush(QPalette::Inactive, QPalette::Light, brush14);
        palette5.setBrush(QPalette::Inactive, QPalette::Midlight, brush15);
        palette5.setBrush(QPalette::Inactive, QPalette::Dark, brush16);
        palette5.setBrush(QPalette::Inactive, QPalette::Text, brush17);
        palette5.setBrush(QPalette::Inactive, QPalette::ButtonText, brush7);
        palette5.setBrush(QPalette::Inactive, QPalette::Base, brush);
        palette5.setBrush(QPalette::Inactive, QPalette::Window, brush18);
        palette5.setBrush(QPalette::Disabled, QPalette::WindowText, brush16);
        palette5.setBrush(QPalette::Disabled, QPalette::Button, brush13);
        palette5.setBrush(QPalette::Disabled, QPalette::Light, brush14);
        palette5.setBrush(QPalette::Disabled, QPalette::Midlight, brush15);
        palette5.setBrush(QPalette::Disabled, QPalette::Dark, brush16);
        palette5.setBrush(QPalette::Disabled, QPalette::Text, brush16);
        palette5.setBrush(QPalette::Disabled, QPalette::ButtonText, brush16);
        palette5.setBrush(QPalette::Disabled, QPalette::Base, brush18);
        palette5.setBrush(QPalette::Disabled, QPalette::Window, brush18);
        totalCurrent->setPalette(palette5);
        totalCurrent->setSmallDecimalPoint(true);
        totalCurrent->setDigitCount(4);
        totalCurrent->setSegmentStyle(QLCDNumber::Filled);
        totalCurrent->setProperty("value", QVariant(8));
        totalCurrent->setProperty("intValue", QVariant(8));

        layoutAmps->addWidget(totalCurrent);

        layoutAmps->setStretch(0, 1);

        layoutValues->addLayout(layoutAmps);

        layoutVolts = new QHBoxLayout();
        layoutVolts->setObjectName(QStringLiteral("layoutVolts"));
        ampsLabel_2 = new QLabel(pwr);
        ampsLabel_2->setObjectName(QStringLiteral("ampsLabel_2"));
        sizePolicy1.setHeightForWidth(ampsLabel_2->sizePolicy().hasHeightForWidth());
        ampsLabel_2->setSizePolicy(sizePolicy1);
        ampsLabel_2->setMaximumSize(QSize(100, 50));
        QPalette palette6;
        palette6.setBrush(QPalette::Active, QPalette::WindowText, brush6);
        palette6.setBrush(QPalette::Inactive, QPalette::WindowText, brush6);
        palette6.setBrush(QPalette::Disabled, QPalette::WindowText, brush12);
        ampsLabel_2->setPalette(palette6);
        ampsLabel_2->setAlignment(Qt::AlignCenter);

        layoutVolts->addWidget(ampsLabel_2);

        averageVoltage = new QLCDNumber(pwr);
        averageVoltage->setObjectName(QStringLiteral("averageVoltage"));
        sizePolicy2.setHeightForWidth(averageVoltage->sizePolicy().hasHeightForWidth());
        averageVoltage->setSizePolicy(sizePolicy2);
        averageVoltage->setMinimumSize(QSize(0, 50));
        averageVoltage->setMaximumSize(QSize(100, 50));
        QPalette palette7;
        QBrush brush19(QColor(24, 224, 224, 255));
        brush19.setStyle(Qt::SolidPattern);
        palette7.setBrush(QPalette::Active, QPalette::WindowText, brush19);
        palette7.setBrush(QPalette::Active, QPalette::Button, brush13);
        palette7.setBrush(QPalette::Active, QPalette::Light, brush14);
        palette7.setBrush(QPalette::Active, QPalette::Midlight, brush15);
        palette7.setBrush(QPalette::Active, QPalette::Dark, brush16);
        palette7.setBrush(QPalette::Active, QPalette::Text, brush17);
        palette7.setBrush(QPalette::Active, QPalette::ButtonText, brush7);
        palette7.setBrush(QPalette::Active, QPalette::Base, brush);
        palette7.setBrush(QPalette::Active, QPalette::Window, brush18);
        palette7.setBrush(QPalette::Inactive, QPalette::WindowText, brush19);
        palette7.setBrush(QPalette::Inactive, QPalette::Button, brush13);
        palette7.setBrush(QPalette::Inactive, QPalette::Light, brush14);
        palette7.setBrush(QPalette::Inactive, QPalette::Midlight, brush15);
        palette7.setBrush(QPalette::Inactive, QPalette::Dark, brush16);
        palette7.setBrush(QPalette::Inactive, QPalette::Text, brush17);
        palette7.setBrush(QPalette::Inactive, QPalette::ButtonText, brush7);
        palette7.setBrush(QPalette::Inactive, QPalette::Base, brush);
        palette7.setBrush(QPalette::Inactive, QPalette::Window, brush18);
        palette7.setBrush(QPalette::Disabled, QPalette::WindowText, brush16);
        palette7.setBrush(QPalette::Disabled, QPalette::Button, brush13);
        palette7.setBrush(QPalette::Disabled, QPalette::Light, brush14);
        palette7.setBrush(QPalette::Disabled, QPalette::Midlight, brush15);
        palette7.setBrush(QPalette::Disabled, QPalette::Dark, brush16);
        palette7.setBrush(QPalette::Disabled, QPalette::Text, brush16);
        palette7.setBrush(QPalette::Disabled, QPalette::ButtonText, brush16);
        palette7.setBrush(QPalette::Disabled, QPalette::Base, brush18);
        palette7.setBrush(QPalette::Disabled, QPalette::Window, brush18);
        averageVoltage->setPalette(palette7);
        averageVoltage->setSmallDecimalPoint(true);
        averageVoltage->setDigitCount(5);
        averageVoltage->setSegmentStyle(QLCDNumber::Filled);
        averageVoltage->setProperty("value", QVariant(119.8));

        layoutVolts->addWidget(averageVoltage);

        layoutVolts->setStretch(0, 1);
        layoutVolts->setStretch(1, 1);

        layoutValues->addLayout(layoutVolts);

        layoutHz = new QHBoxLayout();
        layoutHz->setObjectName(QStringLiteral("layoutHz"));
        ampsLabel_3 = new QLabel(pwr);
        ampsLabel_3->setObjectName(QStringLiteral("ampsLabel_3"));
        sizePolicy1.setHeightForWidth(ampsLabel_3->sizePolicy().hasHeightForWidth());
        ampsLabel_3->setSizePolicy(sizePolicy1);
        ampsLabel_3->setMaximumSize(QSize(100, 50));
        QPalette palette8;
        palette8.setBrush(QPalette::Active, QPalette::WindowText, brush6);
        palette8.setBrush(QPalette::Inactive, QPalette::WindowText, brush6);
        palette8.setBrush(QPalette::Disabled, QPalette::WindowText, brush12);
        ampsLabel_3->setPalette(palette8);
        ampsLabel_3->setAlignment(Qt::AlignCenter);

        layoutHz->addWidget(ampsLabel_3);

        averageFrequency = new QLCDNumber(pwr);
        averageFrequency->setObjectName(QStringLiteral("averageFrequency"));
        QSizePolicy sizePolicy3(QSizePolicy::Minimum, QSizePolicy::Maximum);
        sizePolicy3.setHorizontalStretch(1);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(averageFrequency->sizePolicy().hasHeightForWidth());
        averageFrequency->setSizePolicy(sizePolicy3);
        averageFrequency->setMinimumSize(QSize(0, 50));
        averageFrequency->setMaximumSize(QSize(100, 50));
        QPalette palette9;
        palette9.setBrush(QPalette::Active, QPalette::WindowText, brush19);
        palette9.setBrush(QPalette::Active, QPalette::Button, brush13);
        palette9.setBrush(QPalette::Active, QPalette::Light, brush14);
        palette9.setBrush(QPalette::Active, QPalette::Midlight, brush15);
        palette9.setBrush(QPalette::Active, QPalette::Dark, brush16);
        palette9.setBrush(QPalette::Active, QPalette::Text, brush17);
        palette9.setBrush(QPalette::Active, QPalette::ButtonText, brush7);
        palette9.setBrush(QPalette::Active, QPalette::Base, brush);
        palette9.setBrush(QPalette::Active, QPalette::Window, brush18);
        palette9.setBrush(QPalette::Inactive, QPalette::WindowText, brush19);
        palette9.setBrush(QPalette::Inactive, QPalette::Button, brush13);
        palette9.setBrush(QPalette::Inactive, QPalette::Light, brush14);
        palette9.setBrush(QPalette::Inactive, QPalette::Midlight, brush15);
        palette9.setBrush(QPalette::Inactive, QPalette::Dark, brush16);
        palette9.setBrush(QPalette::Inactive, QPalette::Text, brush17);
        palette9.setBrush(QPalette::Inactive, QPalette::ButtonText, brush7);
        palette9.setBrush(QPalette::Inactive, QPalette::Base, brush);
        palette9.setBrush(QPalette::Inactive, QPalette::Window, brush18);
        palette9.setBrush(QPalette::Disabled, QPalette::WindowText, brush16);
        palette9.setBrush(QPalette::Disabled, QPalette::Button, brush13);
        palette9.setBrush(QPalette::Disabled, QPalette::Light, brush14);
        palette9.setBrush(QPalette::Disabled, QPalette::Midlight, brush15);
        palette9.setBrush(QPalette::Disabled, QPalette::Dark, brush16);
        palette9.setBrush(QPalette::Disabled, QPalette::Text, brush16);
        palette9.setBrush(QPalette::Disabled, QPalette::ButtonText, brush16);
        palette9.setBrush(QPalette::Disabled, QPalette::Base, brush18);
        palette9.setBrush(QPalette::Disabled, QPalette::Window, brush18);
        averageFrequency->setPalette(palette9);
        averageFrequency->setSmallDecimalPoint(true);
        averageFrequency->setDigitCount(4);
        averageFrequency->setSegmentStyle(QLCDNumber::Filled);
        averageFrequency->setProperty("value", QVariant(60));
        averageFrequency->setProperty("intValue", QVariant(60));

        layoutHz->addWidget(averageFrequency);

        layoutHz->setStretch(0, 1);
        layoutHz->setStretch(1, 1);

        layoutValues->addLayout(layoutHz);

        layoutValues->setStretch(0, 1);
        layoutValues->setStretch(1, 1);
        layoutValues->setStretch(2, 1);

        verticalLayout->addLayout(layoutValues);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);

        switchGrid = new QGridLayout();
        switchGrid->setObjectName(QStringLiteral("switchGrid"));
        switchGrid->setSizeConstraint(QLayout::SetDefaultConstraint);
        switchGrid->setContentsMargins(-1, -1, 6, -1);

        verticalLayout->addLayout(switchGrid);

        verticalLayout->setStretch(0, 5);
        verticalLayout->setStretch(1, 1);
        verticalLayout->setStretch(2, 1);
        verticalLayout->setStretch(3, 8);

        gridLayout_2->addLayout(verticalLayout, 0, 0, 1, 1);


        retranslateUi(pwr);

        QMetaObject::connectSlotsByName(pwr);
    } // setupUi

    void retranslateUi(QWidget *pwr)
    {
        pwr->setWindowTitle(QApplication::translate("pwr", "Form", Q_NULLPTR));
        ampsLabel->setText(QApplication::translate("pwr", "Tot. Amps:", Q_NULLPTR));
        ampsLabel_2->setText(QApplication::translate("pwr", "Avg. Volts:", Q_NULLPTR));
        ampsLabel_3->setText(QApplication::translate("pwr", "Avg. Hz:", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class pwr: public Ui_pwr {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_PWR_H
