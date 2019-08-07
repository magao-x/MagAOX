/********************************************************************************
** Form generated from reading UI file 'dmMode.ui'
**
** Created by: Qt User Interface Compiler version 5.9.5
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
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include "qwt_slider.h"

QT_BEGIN_NAMESPACE

class Ui_dmMode
{
public:
    QLabel *title;
    QLabel *modeCurrent_0;
    QLabel *channel;
    QwtSlider *modeSlider_0;
    QLabel *modeName_0;
    QLabel *modeName_2;
    QwtSlider *modeSlider_2;
    QLabel *modeCurrent_2;
    QwtSlider *modeSlider_3;
    QLabel *modeName_3;
    QLabel *modeCurrent_3;
    QwtSlider *modeSlider_4;
    QLabel *modeName_4;
    QLabel *modeCurrent_4;
    QLabel *modeName_5;
    QLabel *modeCurrent_5;
    QwtSlider *modeSlider_5;
    QLabel *modeCurrent_6;
    QLabel *modeName_6;
    QwtSlider *modeSlider_6;
    QLabel *modeCurrent_7;
    QwtSlider *modeSlider_7;
    QLabel *modeName_7;
    QLabel *modeName_8;
    QLabel *modeCurrent_8;
    QwtSlider *modeSlider_8;
    QLabel *modeName_9;
    QLabel *modeCurrent_9;
    QwtSlider *modeSlider_9;
    QLabel *modeCurrent_10;
    QwtSlider *modeSlider_10;
    QLabel *modeName_10;
    QLabel *modeCurrent_1;
    QwtSlider *modeSlider_1;
    QLabel *modeName_1;
    QLineEdit *modeTarget_0;
    QLineEdit *modeTarget_1;
    QLineEdit *modeTarget_2;
    QLineEdit *modeTarget_3;
    QLineEdit *modeTarget_4;
    QLineEdit *modeTarget_5;
    QLineEdit *modeTarget_6;
    QLineEdit *modeTarget_7;
    QLineEdit *modeTarget_8;
    QLineEdit *modeTarget_9;
    QLineEdit *modeTarget_10;

    void setupUi(QDialog *dmMode)
    {
        if (dmMode->objectName().isEmpty())
            dmMode->setObjectName(QStringLiteral("dmMode"));
        dmMode->resize(621, 730);
        title = new QLabel(dmMode);
        title->setObjectName(QStringLiteral("title"));
        title->setGeometry(QRect(30, 10, 561, 24));
        title->setAlignment(Qt::AlignCenter);
        modeCurrent_0 = new QLabel(dmMode);
        modeCurrent_0->setObjectName(QStringLiteral("modeCurrent_0"));
        modeCurrent_0->setGeometry(QRect(550, 95, 61, 41));
        modeCurrent_0->setFrameShape(QFrame::Panel);
        modeCurrent_0->setFrameShadow(QFrame::Sunken);
        modeCurrent_0->setTextFormat(Qt::PlainText);
        channel = new QLabel(dmMode);
        channel->setObjectName(QStringLiteral("channel"));
        channel->setGeometry(QRect(30, 50, 561, 24));
        channel->setAlignment(Qt::AlignCenter);
        modeSlider_0 = new QwtSlider(dmMode);
        modeSlider_0->setObjectName(QStringLiteral("modeSlider_0"));
        modeSlider_0->setGeometry(QRect(120, 105, 341, 21));
        modeSlider_0->setLowerBound(-1);
        modeSlider_0->setUpperBound(1);
        modeSlider_0->setOrientation(Qt::Horizontal);
        modeSlider_0->setScalePosition(QwtSlider::NoScale);
        modeName_0 = new QLabel(dmMode);
        modeName_0->setObjectName(QStringLiteral("modeName_0"));
        modeName_0->setGeometry(QRect(5, 100, 91, 24));
        modeName_2 = new QLabel(dmMode);
        modeName_2->setObjectName(QStringLiteral("modeName_2"));
        modeName_2->setGeometry(QRect(5, 210, 91, 24));
        modeSlider_2 = new QwtSlider(dmMode);
        modeSlider_2->setObjectName(QStringLiteral("modeSlider_2"));
        modeSlider_2->setGeometry(QRect(120, 215, 341, 21));
        modeSlider_2->setLowerBound(-1);
        modeSlider_2->setUpperBound(1);
        modeSlider_2->setOrientation(Qt::Horizontal);
        modeSlider_2->setScalePosition(QwtSlider::NoScale);
        modeCurrent_2 = new QLabel(dmMode);
        modeCurrent_2->setObjectName(QStringLiteral("modeCurrent_2"));
        modeCurrent_2->setGeometry(QRect(550, 205, 61, 41));
        modeCurrent_2->setFrameShape(QFrame::Panel);
        modeCurrent_2->setFrameShadow(QFrame::Sunken);
        modeCurrent_2->setTextFormat(Qt::PlainText);
        modeSlider_3 = new QwtSlider(dmMode);
        modeSlider_3->setObjectName(QStringLiteral("modeSlider_3"));
        modeSlider_3->setGeometry(QRect(120, 265, 341, 21));
        modeSlider_3->setLowerBound(-1);
        modeSlider_3->setUpperBound(1);
        modeSlider_3->setOrientation(Qt::Horizontal);
        modeSlider_3->setScalePosition(QwtSlider::NoScale);
        modeName_3 = new QLabel(dmMode);
        modeName_3->setObjectName(QStringLiteral("modeName_3"));
        modeName_3->setGeometry(QRect(5, 260, 91, 24));
        modeCurrent_3 = new QLabel(dmMode);
        modeCurrent_3->setObjectName(QStringLiteral("modeCurrent_3"));
        modeCurrent_3->setGeometry(QRect(550, 255, 61, 41));
        modeCurrent_3->setFrameShape(QFrame::Panel);
        modeCurrent_3->setFrameShadow(QFrame::Sunken);
        modeCurrent_3->setTextFormat(Qt::PlainText);
        modeSlider_4 = new QwtSlider(dmMode);
        modeSlider_4->setObjectName(QStringLiteral("modeSlider_4"));
        modeSlider_4->setGeometry(QRect(120, 315, 341, 21));
        modeSlider_4->setLowerBound(-1);
        modeSlider_4->setUpperBound(1);
        modeSlider_4->setOrientation(Qt::Horizontal);
        modeSlider_4->setScalePosition(QwtSlider::NoScale);
        modeName_4 = new QLabel(dmMode);
        modeName_4->setObjectName(QStringLiteral("modeName_4"));
        modeName_4->setGeometry(QRect(5, 310, 91, 24));
        modeCurrent_4 = new QLabel(dmMode);
        modeCurrent_4->setObjectName(QStringLiteral("modeCurrent_4"));
        modeCurrent_4->setGeometry(QRect(550, 305, 61, 41));
        modeCurrent_4->setFrameShape(QFrame::Panel);
        modeCurrent_4->setFrameShadow(QFrame::Sunken);
        modeCurrent_4->setTextFormat(Qt::PlainText);
        modeName_5 = new QLabel(dmMode);
        modeName_5->setObjectName(QStringLiteral("modeName_5"));
        modeName_5->setGeometry(QRect(5, 360, 91, 24));
        modeCurrent_5 = new QLabel(dmMode);
        modeCurrent_5->setObjectName(QStringLiteral("modeCurrent_5"));
        modeCurrent_5->setGeometry(QRect(550, 355, 61, 41));
        modeCurrent_5->setFrameShape(QFrame::Panel);
        modeCurrent_5->setFrameShadow(QFrame::Sunken);
        modeCurrent_5->setTextFormat(Qt::PlainText);
        modeSlider_5 = new QwtSlider(dmMode);
        modeSlider_5->setObjectName(QStringLiteral("modeSlider_5"));
        modeSlider_5->setGeometry(QRect(120, 365, 341, 21));
        modeSlider_5->setLowerBound(-1);
        modeSlider_5->setUpperBound(1);
        modeSlider_5->setOrientation(Qt::Horizontal);
        modeSlider_5->setScalePosition(QwtSlider::NoScale);
        modeCurrent_6 = new QLabel(dmMode);
        modeCurrent_6->setObjectName(QStringLiteral("modeCurrent_6"));
        modeCurrent_6->setGeometry(QRect(550, 405, 61, 41));
        modeCurrent_6->setFrameShape(QFrame::Panel);
        modeCurrent_6->setFrameShadow(QFrame::Sunken);
        modeCurrent_6->setTextFormat(Qt::PlainText);
        modeName_6 = new QLabel(dmMode);
        modeName_6->setObjectName(QStringLiteral("modeName_6"));
        modeName_6->setGeometry(QRect(5, 410, 91, 24));
        modeSlider_6 = new QwtSlider(dmMode);
        modeSlider_6->setObjectName(QStringLiteral("modeSlider_6"));
        modeSlider_6->setGeometry(QRect(120, 415, 341, 21));
        modeSlider_6->setLowerBound(-1);
        modeSlider_6->setUpperBound(1);
        modeSlider_6->setOrientation(Qt::Horizontal);
        modeSlider_6->setScalePosition(QwtSlider::NoScale);
        modeCurrent_7 = new QLabel(dmMode);
        modeCurrent_7->setObjectName(QStringLiteral("modeCurrent_7"));
        modeCurrent_7->setGeometry(QRect(550, 455, 61, 41));
        modeCurrent_7->setFrameShape(QFrame::Panel);
        modeCurrent_7->setFrameShadow(QFrame::Sunken);
        modeCurrent_7->setTextFormat(Qt::PlainText);
        modeSlider_7 = new QwtSlider(dmMode);
        modeSlider_7->setObjectName(QStringLiteral("modeSlider_7"));
        modeSlider_7->setGeometry(QRect(120, 465, 341, 21));
        modeSlider_7->setLowerBound(-1);
        modeSlider_7->setUpperBound(1);
        modeSlider_7->setOrientation(Qt::Horizontal);
        modeSlider_7->setScalePosition(QwtSlider::NoScale);
        modeName_7 = new QLabel(dmMode);
        modeName_7->setObjectName(QStringLiteral("modeName_7"));
        modeName_7->setGeometry(QRect(5, 460, 91, 24));
        modeName_8 = new QLabel(dmMode);
        modeName_8->setObjectName(QStringLiteral("modeName_8"));
        modeName_8->setGeometry(QRect(5, 510, 91, 24));
        modeCurrent_8 = new QLabel(dmMode);
        modeCurrent_8->setObjectName(QStringLiteral("modeCurrent_8"));
        modeCurrent_8->setGeometry(QRect(550, 505, 61, 41));
        modeCurrent_8->setFrameShape(QFrame::Panel);
        modeCurrent_8->setFrameShadow(QFrame::Sunken);
        modeCurrent_8->setTextFormat(Qt::PlainText);
        modeSlider_8 = new QwtSlider(dmMode);
        modeSlider_8->setObjectName(QStringLiteral("modeSlider_8"));
        modeSlider_8->setGeometry(QRect(120, 515, 341, 21));
        modeSlider_8->setLowerBound(-1);
        modeSlider_8->setUpperBound(1);
        modeSlider_8->setOrientation(Qt::Horizontal);
        modeSlider_8->setScalePosition(QwtSlider::NoScale);
        modeName_9 = new QLabel(dmMode);
        modeName_9->setObjectName(QStringLiteral("modeName_9"));
        modeName_9->setGeometry(QRect(5, 561, 91, 24));
        modeCurrent_9 = new QLabel(dmMode);
        modeCurrent_9->setObjectName(QStringLiteral("modeCurrent_9"));
        modeCurrent_9->setGeometry(QRect(550, 556, 61, 41));
        modeCurrent_9->setFrameShape(QFrame::Panel);
        modeCurrent_9->setFrameShadow(QFrame::Sunken);
        modeCurrent_9->setTextFormat(Qt::PlainText);
        modeSlider_9 = new QwtSlider(dmMode);
        modeSlider_9->setObjectName(QStringLiteral("modeSlider_9"));
        modeSlider_9->setGeometry(QRect(120, 566, 341, 21));
        modeSlider_9->setLowerBound(-1);
        modeSlider_9->setUpperBound(1);
        modeSlider_9->setOrientation(Qt::Horizontal);
        modeSlider_9->setScalePosition(QwtSlider::NoScale);
        modeCurrent_10 = new QLabel(dmMode);
        modeCurrent_10->setObjectName(QStringLiteral("modeCurrent_10"));
        modeCurrent_10->setGeometry(QRect(550, 605, 61, 41));
        modeCurrent_10->setFrameShape(QFrame::Panel);
        modeCurrent_10->setFrameShadow(QFrame::Sunken);
        modeCurrent_10->setTextFormat(Qt::PlainText);
        modeSlider_10 = new QwtSlider(dmMode);
        modeSlider_10->setObjectName(QStringLiteral("modeSlider_10"));
        modeSlider_10->setGeometry(QRect(120, 615, 341, 21));
        modeSlider_10->setLowerBound(-1);
        modeSlider_10->setUpperBound(1);
        modeSlider_10->setOrientation(Qt::Horizontal);
        modeSlider_10->setScalePosition(QwtSlider::NoScale);
        modeName_10 = new QLabel(dmMode);
        modeName_10->setObjectName(QStringLiteral("modeName_10"));
        modeName_10->setGeometry(QRect(5, 610, 91, 24));
        modeCurrent_1 = new QLabel(dmMode);
        modeCurrent_1->setObjectName(QStringLiteral("modeCurrent_1"));
        modeCurrent_1->setGeometry(QRect(550, 150, 61, 41));
        modeCurrent_1->setFrameShape(QFrame::Panel);
        modeCurrent_1->setFrameShadow(QFrame::Sunken);
        modeCurrent_1->setTextFormat(Qt::PlainText);
        modeSlider_1 = new QwtSlider(dmMode);
        modeSlider_1->setObjectName(QStringLiteral("modeSlider_1"));
        modeSlider_1->setGeometry(QRect(120, 160, 341, 21));
        modeSlider_1->setLowerBound(-1);
        modeSlider_1->setUpperBound(1);
        modeSlider_1->setOrientation(Qt::Horizontal);
        modeSlider_1->setScalePosition(QwtSlider::NoScale);
        modeName_1 = new QLabel(dmMode);
        modeName_1->setObjectName(QStringLiteral("modeName_1"));
        modeName_1->setGeometry(QRect(5, 155, 91, 24));
        modeTarget_0 = new QLineEdit(dmMode);
        modeTarget_0->setObjectName(QStringLiteral("modeTarget_0"));
        modeTarget_0->setGeometry(QRect(469, 95, 67, 41));
        modeTarget_1 = new QLineEdit(dmMode);
        modeTarget_1->setObjectName(QStringLiteral("modeTarget_1"));
        modeTarget_1->setGeometry(QRect(469, 150, 67, 41));
        modeTarget_2 = new QLineEdit(dmMode);
        modeTarget_2->setObjectName(QStringLiteral("modeTarget_2"));
        modeTarget_2->setGeometry(QRect(469, 205, 67, 41));
        modeTarget_3 = new QLineEdit(dmMode);
        modeTarget_3->setObjectName(QStringLiteral("modeTarget_3"));
        modeTarget_3->setGeometry(QRect(470, 260, 67, 41));
        modeTarget_4 = new QLineEdit(dmMode);
        modeTarget_4->setObjectName(QStringLiteral("modeTarget_4"));
        modeTarget_4->setGeometry(QRect(469, 310, 67, 41));
        modeTarget_5 = new QLineEdit(dmMode);
        modeTarget_5->setObjectName(QStringLiteral("modeTarget_5"));
        modeTarget_5->setGeometry(QRect(468, 360, 67, 41));
        modeTarget_6 = new QLineEdit(dmMode);
        modeTarget_6->setObjectName(QStringLiteral("modeTarget_6"));
        modeTarget_6->setGeometry(QRect(467, 410, 67, 41));
        modeTarget_7 = new QLineEdit(dmMode);
        modeTarget_7->setObjectName(QStringLiteral("modeTarget_7"));
        modeTarget_7->setGeometry(QRect(467, 455, 67, 41));
        modeTarget_8 = new QLineEdit(dmMode);
        modeTarget_8->setObjectName(QStringLiteral("modeTarget_8"));
        modeTarget_8->setGeometry(QRect(466, 505, 67, 41));
        modeTarget_9 = new QLineEdit(dmMode);
        modeTarget_9->setObjectName(QStringLiteral("modeTarget_9"));
        modeTarget_9->setGeometry(QRect(467, 555, 67, 41));
        modeTarget_10 = new QLineEdit(dmMode);
        modeTarget_10->setObjectName(QStringLiteral("modeTarget_10"));
        modeTarget_10->setGeometry(QRect(466, 605, 67, 41));

        retranslateUi(dmMode);

        QMetaObject::connectSlotsByName(dmMode);
    } // setupUi

    void retranslateUi(QDialog *dmMode)
    {
        dmMode->setWindowTitle(QApplication::translate("dmMode", "Dialog", Q_NULLPTR));
        title->setText(QApplication::translate("dmMode", "DM Modes", Q_NULLPTR));
        modeCurrent_0->setText(QApplication::translate("dmMode", "0.0", Q_NULLPTR));
        channel->setText(QApplication::translate("dmMode", "dmXXdispYY", Q_NULLPTR));
        modeName_0->setText(QApplication::translate("dmMode", "Mode 00", Q_NULLPTR));
        modeName_2->setText(QApplication::translate("dmMode", "Mode 02", Q_NULLPTR));
        modeCurrent_2->setText(QApplication::translate("dmMode", "0.0", Q_NULLPTR));
        modeName_3->setText(QApplication::translate("dmMode", "Mode 03", Q_NULLPTR));
        modeCurrent_3->setText(QApplication::translate("dmMode", "0.0", Q_NULLPTR));
        modeName_4->setText(QApplication::translate("dmMode", "Mode 04", Q_NULLPTR));
        modeCurrent_4->setText(QApplication::translate("dmMode", "0.0", Q_NULLPTR));
        modeName_5->setText(QApplication::translate("dmMode", "Mode 05", Q_NULLPTR));
        modeCurrent_5->setText(QApplication::translate("dmMode", "0.0", Q_NULLPTR));
        modeCurrent_6->setText(QApplication::translate("dmMode", "0.0", Q_NULLPTR));
        modeName_6->setText(QApplication::translate("dmMode", "Mode 06", Q_NULLPTR));
        modeCurrent_7->setText(QApplication::translate("dmMode", "0.0", Q_NULLPTR));
        modeName_7->setText(QApplication::translate("dmMode", "Mode 07", Q_NULLPTR));
        modeName_8->setText(QApplication::translate("dmMode", "Mode 08", Q_NULLPTR));
        modeCurrent_8->setText(QApplication::translate("dmMode", "0.0", Q_NULLPTR));
        modeName_9->setText(QApplication::translate("dmMode", "Mode 09", Q_NULLPTR));
        modeCurrent_9->setText(QApplication::translate("dmMode", "0.0", Q_NULLPTR));
        modeCurrent_10->setText(QApplication::translate("dmMode", "0.0", Q_NULLPTR));
        modeName_10->setText(QApplication::translate("dmMode", "Mode 10", Q_NULLPTR));
        modeCurrent_1->setText(QApplication::translate("dmMode", "0.0", Q_NULLPTR));
        modeName_1->setText(QApplication::translate("dmMode", "Mode 01", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class dmMode: public Ui_dmMode {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DMMODE_H
