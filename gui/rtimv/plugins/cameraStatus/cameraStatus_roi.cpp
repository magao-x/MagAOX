/** \file cameraStatus_roi.cpp
 * \brief Coherent ROI editing and explicit Check/Set authorization on the GUI thread.
 */
#include "cameraStatus.hpp"

#include <QSignalBlocker>
#include <cmath>
#include <limits>

namespace
{
/// Read six complete, finite native coordinates without truncating integer fields.
bool readROI(const rtimvPropertySnapshot &property, /**< [in] Complete property message. */
             const QString &prefix, /**< [in] Prefix of the six element names. */
             MagAOX::roi::region &roi /**< [out] Parsed region. */)
{
    if(!property.m_valid || !property.m_complete || property.m_type != rtimvPropertyType::number) return false;
    std::vector<double> values;
    for(const auto &name : MagAOX::roi::names(prefix.toStdString()))
    {
        bool ok;
        double value = property.m_elements.value(QString::fromStdString(name)).toDouble(&ok);
        if(!ok || !std::isfinite(value)) return false;
        if(values.size() >= 2 && (value <= 0 || value > std::numeric_limits<int>::max() || value != std::floor(value))) return false;
        if(values.size() < 2 && std::abs(value) > std::numeric_limits<float>::max()) return false;
        values.push_back(value);
    }
    roi = {static_cast<float>(values[0]), static_cast<float>(values[1]), static_cast<int>(values[2]),
           static_cast<int>(values[3]), static_cast<int>(values[4]), static_cast<int>(values[5])};
    return MagAOX::roi::valid(roi);
}

/// Compare centers at their native float precision, and every integer field exactly.
bool sameROI(const MagAOX::roi::region &a, /**< [in] First native region. */
             const MagAOX::roi::region &b /**< [in] Second native region. */)
{
    // INDI serializes float centers as doubles. Parsing back to native floats needs no display tolerance.
    return MagAOX::roi::same(a, b);
}

/// Construct one camera request switch, preserving its boolean type.
rtimvCommand requestROI(const std::string &device, /**< [in] Camera device. */
                        const QString &property /**< [in] Request property. */)
{
    return {QString::fromStdString(device), property, rtimvPropertyType::toggle, {{"request", true}}};
}
}

rtimvPropertySnapshot cameraStatus::roiProperty(const QString &name)
{
    if(!m_commands && m_roa.m_mainWindowObject)
        m_commands = m_roa.m_mainWindowObject->findChild<rtimvCommandRouter *>("rtimv.commandRouter");
    return m_commands ? m_commands->snapshot("indi", QString::fromStdString(m_deviceName), name) : rtimvPropertySnapshot{};
}

bool cameraStatus::updateROI()
{
    auto state = roiProperty("roi_region_state");
    m_commandSeen = m_commandSeen || state.m_valid;
    roiT current, target;
    bool geometry = readROI(state, "", current) && readROI(state, "target_", target);
    bool unitsOK;
    int units = state.m_elements.value("units").toInt(&unitsOK);
    geometry = geometry && unitsOK && (units == 0 || units == 1);
    const auto policy = static_cast<MagAOX::roi::units>(units);
    auto completed = roiProperty("roi_check_result");
    auto check = roiProperty("roi_region_check"), set = roiProperty("roi_set");
    bool capable = state.m_valid && completed.m_valid && check.m_valid && check.m_writable &&
                   check.m_type == rtimvPropertyType::toggle && check.m_elements.contains("request") &&
                   set.m_valid && set.m_writable && set.m_type == rtimvPropertyType::toggle && set.m_elements.contains("request");
    for(const auto &name : {"x", "y", "w", "h"})
    {
        auto property = roiProperty(QString("roi_region_") + name);
        capable = capable && property.m_valid && property.m_writable && property.m_type == rtimvPropertyType::number &&
                  property.m_generation == state.m_generation && property.m_elements.contains("target");
    }
    capable = capable && completed.m_generation == state.m_generation && check.m_generation == state.m_generation &&
              set.m_generation == state.m_generation;
    QSize image;
    if(m_roa.m_mainWindowObject)
        QMetaObject::invokeMethod(m_roa.m_mainWindowObject, "overlayImageSize", Qt::DirectConnection, Q_RETURN_ARG(QSize, image));
    const bool imageMatches = geometry && image.width() == MagAOX::roi::imageSize(current.w, current.bin_x, policy) &&
                              image.height() == MagAOX::roi::imageSize(current.h, current.bin_y, policy);
    m_roiReady = geometry && capable && imageMatches && state.m_state == "ok";

    if(m_roiPhase == roiPhase::applying)
    {
        // Configuring and a delayed image resize are expected after Set. Do not revoke a submitted operation.
        if(!capable || state.m_generation != m_roiGeneration || set.m_state == "alert")
            failROI("ROI apply lost or failed");
        else if(geometry && state.m_state == "ok" && !sameROI(current, m_checkCurrent))
        {
            const bool adjusted = !sameROI(current, m_requestedROI);
            m_roiPhase = roiPhase::idle;
            m_explicitROI = false;
            emit roiMessage(adjusted ? "Applied ROI differs from Check" : "ROI applied");
        }
        else if(m_roiWait.hasExpired(m_roiTimeout)) failROI("ROI apply timed out");
        else
        {
            if(m_roiBox) { m_roiBox->setStretchable(false); m_roiBox->setVisible(imageMatches && m_explicitROI); }
            return true;
        }
    }

    if(!m_roiReady)
    {
        if(m_roiPhase != roiPhase::idle) failROI("ROI unavailable; review target");
        if(m_roiBox && m_commandSeen) { m_roiBox->emitDeSelected(); m_roiBox->setVisible(false); m_roiBox->setStretchable(false); }
        return m_commandSeen;
    }

    const bool newReference = m_roiGeneration != state.m_generation || m_roiUnits != policy || !sameROI(m_currentROI, current);
    const bool newTarget = !sameROI(m_targetROI, target);
    if(newReference && m_roiPhase != roiPhase::idle) failROI("ROI reference changed");
    else if(newTarget && (m_roiPhase == roiPhase::editing || m_roiPhase == roiPhase::dragging || m_roiPhase == roiPhase::adjusted))
        failROI("ROI target changed externally");
    m_roiGeneration = state.m_generation;
    m_roiUnits = policy;
    m_currentROI = current;
    m_targetROI = target;
    m_newBoxPreset = state.m_elements.value("preset_known").toBool() &&
                    (state.m_elements.value("is_full").toBool() || state.m_elements.value("is_default").toBool());

    if(m_roiPhase == roiPhase::awaitingTarget)
    {
        if(sameROI(target, m_desiredROI))
        {
            m_roiPhase = roiPhase::editing;
            const bool apply = m_applyAfterEcho;
            m_applyAfterEcho = false;
            if(apply) applyROI();
        }
        else
        {
            auto values = MagAOX::roi::values(target), old = MagAOX::roi::values(m_echoBase), desired = MagAOX::roi::values(m_desiredROI);
            bool expected = true;
            for(size_t n = 0; n < values.size(); ++n) expected = expected && (values[n] == old[n] || values[n] == desired[n]);
            if(!expected) failROI("ROI target changed externally");
            else if(m_roiWait.hasExpired(m_roiTimeout)) failROI("ROI target echo timed out");
            else return true; // Keep the final local rectangle while the ordered axis echoes arrive.
        }
    }
    if(m_roiPhase == roiPhase::checking)
    {
        bool sequenceOK;
        const double sequence = completed.m_elements.value("sequence").toDouble(&sequenceOK);
        if(sequenceOK && std::isfinite(sequence) && sequence > m_checkSequence && sequence < 9007199254740992.0 && sequence == std::floor(sequence))
        {
            roiT requested, checked, reference;
            const bool matches = completed.m_state == "ok" && readROI(completed, "requested_", requested) &&
                                 readROI(completed, "checked_", checked) && readROI(completed, "current_", reference) &&
                                 sameROI(requested, m_requestedROI) && sameROI(reference, m_checkCurrent) && sameROI(target, checked);
            if(!matches) failROI("ROI Check failed or changed");
            else if(checked.bin_x != requested.bin_x || checked.bin_y != requested.bin_y)
                failROI("Binning adjusted; use roiGUI");
            else if(!sameROI(checked, requested))
            {
                m_roiPhase = roiPhase::adjusted;
                m_explicitROI = true;
                emit roiMessage("ROI adjusted; Alt+R to apply");
            }
            else if(sameROI(checked, current))
            {
                m_roiPhase = roiPhase::idle;
                m_explicitROI = false;
                emit roiMessage("ROI unchanged");
            }
            else
            {
                m_roiPhase = roiPhase::applying;
                m_roiWait.start();
                sendROI({requestROI(m_deviceName, "roi_set")});
            }
        }
        else if(m_roiWait.hasExpired(m_roiTimeout)) failROI("ROI Check timed out");
    }
    if(m_roiPhase != roiPhase::dragging) drawROI();
    return true;
}

void cameraStatus::makeROIBox()
{
    if(m_roiBox) return;
    m_roiBox = new StretchBox;
    m_roiBox->setPenColor(Qt::magenta);
    m_roiBox->setPenWidth(1);
    m_roiBox->setRemovable(false);
    m_roiBox->setMaintainCenter(true);
    m_roiBox->setStretchable(false);
    connect(m_roiBox, &StretchBox::editStarted, this, &cameraStatus::roiEditStarted);
    connect(m_roiBox, &StretchBox::editFinished, this, &cameraStatus::roiEditFinished);
    connect(m_roiBox, QOverload<StretchBox *>::of(&StretchBox::remove), this, &cameraStatus::stretchBoxRemove);
    // Old rtimv hosts still support the passive preview, but not this optional editing integration.
    if(m_roa.m_mainWindowObject && m_roa.m_mainWindowObject->metaObject()->indexOfSlot("addInteractiveBox(StretchBox*)") >= 0)
        emit newInteractiveBox(m_roiBox);
    else emit newStretchBox(m_roiBox);
}

QRectF cameraStatus::roiRect() const
{
    using namespace MagAOX::roi;
    const auto &c = m_currentROI, &t = m_targetROI;
    double x = relative(t.x, t.bin_x, c.x, c.w, c.bin_x, m_roiUnits) + 0.5;
    double y = imageSize(c.h, c.bin_y, m_roiUnits) - relative(t.y, t.bin_y, c.y, c.h, c.bin_y, m_roiUnits) - 0.5;
    double w = static_cast<double>(t.w) * (m_roiUnits == units::binned ? t.bin_x : 1) / c.bin_x;
    double h = static_cast<double>(t.h) * (m_roiUnits == units::binned ? t.bin_y : 1) / c.bin_y;
    return QRectF(x-w/2, y-h/2, w, h);
}

void cameraStatus::drawROI()
{
    makeROIBox();
    QSignalBlocker blocker(m_roiBox);
    m_roiBox->setPos(0, 0);
    m_roiBox->setRect(roiRect());
    m_roiBox->setVisible(m_explicitROI || !sameROI(m_currentROI, m_targetROI));
    m_roiBox->setStretchable(m_roiReady && (m_roiPhase == roiPhase::editing || m_roiPhase == roiPhase::adjusted));
    if(!m_roiBox->isVisible()) { blocker.unblock(); m_roiBox->emitDeSelected(); }
}

void cameraStatus::beginROI()
{
    updateROI();
    if(!m_roiReady) { emit roiMessage("ROI commands unavailable"); return; }
    if(m_explicitROI) { if(m_roiBox) { m_roiBox->setVisible(true); m_roiBox->emitSelected(); } return; }
    if(m_roiPhase != roiPhase::idle) return;
    m_explicitROI = true;
    m_roiPhase = roiPhase::editing;
    drawROI();
    if(sameROI(m_targetROI, m_currentROI) && m_newBoxPreset)
    {
        QRectF rect;
        if(!QMetaObject::invokeMethod(m_roa.m_mainWindowObject, "newUserBoxRect", Qt::DirectConnection, Q_RETURN_ARG(QRectF, rect)) || !rect.isValid())
        { failROI("ROI placement unavailable"); return; }
        stageROI(rect, m_targetROI, true);
    }
    m_roiBox->setVisible(true);
    m_roiBox->setSelected(true);
    m_roiBox->setFocus();
    m_roiBox->emitSelected();
}

void cameraStatus::roiEditStarted(StretchBox *box)
{
    if(box != m_roiBox || !m_roiReady || (m_roiPhase != roiPhase::editing && m_roiPhase != roiPhase::adjusted)) return;
    m_dragROI = m_targetROI;
    m_dragRect = box->mapRectToScene(box->rect());
    m_roiPhase = roiPhase::dragging;
}

void cameraStatus::roiEditFinished(StretchBox *box)
{
    if(box != m_roiBox || m_roiPhase != roiPhase::dragging) return;
    const QRectF rect = box->mapRectToScene(box->rect());
    updateROI(); // Detect a changed current/reference or an external target before staging.
    if(m_roiPhase != roiPhase::dragging) return;
    m_roiPhase = roiPhase::editing;
    stageROI(rect, m_dragROI, rect.size() != m_dragRect.size());
}

bool cameraStatus::stageROI(const QRectF &rect, const roiT &before, bool resized)
{
    using namespace MagAOX::roi;
    roiT target = before;
    const auto &c = m_currentROI;
    double x = absolute(rect.center().x()-.5, target.bin_x, c.x, c.w, c.bin_x, m_roiUnits);
    double y = absolute(imageSize(c.h, c.bin_y, m_roiUnits)-rect.center().y()-.5, target.bin_y, c.y, c.h, c.bin_y, m_roiUnits);
    if(!rect.isValid() || !std::isfinite(x) || !std::isfinite(y) || std::abs(x) > std::numeric_limits<float>::max() || std::abs(y) > std::numeric_limits<float>::max())
    { failROI("Invalid ROI rectangle"); return false; }
    target.x = x;
    target.y = y;
    if(resized)
    {
        double w = std::round(rect.width()*c.bin_x/(m_roiUnits == units::binned ? target.bin_x : 1));
        double h = std::round(rect.height()*c.bin_y/(m_roiUnits == units::binned ? target.bin_y : 1));
        if(!std::isfinite(w) || !std::isfinite(h) || w < 1 || h < 1 || w > std::numeric_limits<int>::max() || h > std::numeric_limits<int>::max())
        { failROI("ROI size is not representable"); return false; }
        target.w = w;
        target.h = h;
    }
    QVector<rtimvCommand> commands;
    auto old = values(before), desired = values(target);
    auto fields = names();
    for(size_t n = 0; n < 4; ++n)
        if(old[n] != desired[n]) commands.push_back({QString::fromStdString(m_deviceName), QString::fromStdString("roi_region_"+fields[n]),
                                                   rtimvPropertyType::number, {{"target", desired[n]}}});
    if(commands.empty()) { drawROI(); return true; }
    m_desiredROI = target;
    m_echoBase = before;
    m_checkCurrent = c;
    m_roiPhase = roiPhase::awaitingTarget;
    m_applyAfterEcho = false;
    m_roiWait.start();
    if(!sendROI(commands)) return false;
    // Show the representable local target, but retain the backend target as the echo baseline.
    m_targetROI = target;
    drawROI();
    m_targetROI = before;
    return true;
}

void cameraStatus::applyROI()
{
    // Do not poll Check completion here: a key arriving before adjusted geometry is drawn cannot confirm it.
    if(m_roiPhase == roiPhase::checking || m_roiPhase == roiPhase::applying) return;
    if(m_roiPhase == roiPhase::awaitingTarget) { if(!m_applyAfterEcho) m_applyAfterEcho = true; return; }
    if(m_roiPhase == roiPhase::dragging && m_roiBox)
    {
        m_roiBox->cancelEdit();
        roiEditFinished(m_roiBox);
        if(m_roiPhase == roiPhase::awaitingTarget) { m_applyAfterEcho = true; return; }
    }
    updateROI();
    if(!m_roiReady || !m_explicitROI || (m_roiPhase != roiPhase::editing && m_roiPhase != roiPhase::adjusted))
    { emit roiMessage("Shift+R to edit a valid ROI"); return; }
    auto completed = roiProperty("roi_check_result");
    bool ok;
    double sequence = completed.m_elements.value("sequence").toDouble(&ok);
    if(!completed.m_complete || !ok || !std::isfinite(sequence) || sequence < 0 || sequence >= 9007199254740992.0 || sequence != std::floor(sequence))
    { failROI("ROI Check status unavailable"); return; }
    m_checkSequence = sequence;
    m_checkCurrent = m_currentROI;
    m_requestedROI = m_targetROI;
    m_roiPhase = roiPhase::checking;
    m_roiWait.start();
    m_roiBox->setStretchable(false);
    sendROI({requestROI(m_deviceName, "roi_region_check")});
}

bool cameraStatus::sendROI(const QVector<rtimvCommand> &commands)
{
    QString error;
    if(m_commands && m_commands->send("indi", commands, error)) return true;
    failROI(error.isEmpty() ? "ROI provider unavailable" : error);
    return false;
}

void cameraStatus::failROI(const QString &message)
{
    if(m_roiBox) { m_roiBox->cancelEdit(); m_roiBox->emitDeSelected(); m_roiBox->setStretchable(false); }
    m_roiPhase = roiPhase::idle;
    m_explicitROI = false;
    m_applyAfterEcho = false;
    emit roiMessage(message);
}

void cameraStatus::escapeROI()
{
    if(!m_explicitROI) return;
    if(m_roiBox) { m_roiBox->cancelEdit(); m_roiBox->emitDeSelected(); }
    m_explicitROI = false;
    m_applyAfterEcho = false;
    if(m_roiPhase == roiPhase::applying) { emit roiMessage("ROI Set already submitted"); return; }
    m_roiPhase = roiPhase::idle;
    if(m_roiReady) drawROI();
}
