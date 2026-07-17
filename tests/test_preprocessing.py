import pytest
import pandas as pd
from src.ElijahA.preprocessing import TypeDummyCreator


@pytest.fixture
def house_df():
    return pd.DataFrame({
        'type': ['Condo', 'Single-family', 'Townhome'],
        'sq_ft': [704, 1732, 1445],
    })


def test_fit_stores_categories(house_df):
    tdc = TypeDummyCreator()
    tdc.fit(house_df)
    assert tdc.categories_ == {'type': ['Condo', 'Single-family', 'Townhome']}


def test_transform_replaces_column_with_dummies(house_df):
    tdc = TypeDummyCreator()
    out = tdc.fit(house_df).transform(house_df)
    assert 'type' not in out.columns
    assert 'type_Condo' in out.columns
    assert 'type_Single-family' in out.columns
    assert 'type_Townhome' in out.columns


def test_dummy_values_are_correct(house_df):
    tdc = TypeDummyCreator()
    out = tdc.fit(house_df).transform(house_df)
    assert out['type_Condo'].tolist() == [1, 0, 0]
    assert out['type_Single-family'].tolist() == [0, 1, 0]
    assert out['type_Townhome'].tolist() == [0, 0, 1]


def test_unseen_category_fills_with_zeros():
    train = pd.DataFrame({'type': ['Condo', 'Single-family']})
    test = pd.DataFrame({'type': ['Townhome']})
    tdc = TypeDummyCreator()
    tdc.fit(train)
    out = tdc.transform(test)
    assert list(out.columns) == ['type_Condo', 'type_Single-family']
    assert out['type_Condo'].iloc[0] == 0
    assert out['type_Single-family'].iloc[0] == 0


def test_consistent_output_columns_when_category_absent():
    train = pd.DataFrame({'type': ['Condo', 'Single-family', 'Townhome']})
    test = pd.DataFrame({'type': ['Condo', 'Condo']})
    tdc = TypeDummyCreator()
    tdc.fit(train)
    out = tdc.transform(test)
    assert set(out.columns) == {'type_Condo', 'type_Single-family', 'type_Townhome'}
    assert out['type_Townhome'].sum() == 0


def test_multiple_columns():
    df = pd.DataFrame({
        'type': ['Condo', 'Single-family'],
        'city': ['Concord', 'Walnut Creek'],
        'sq_ft': [700, 1500],
    })
    tdc = TypeDummyCreator(columns=['type', 'city'])
    out = tdc.fit(df).transform(df)
    assert 'type' not in out.columns
    assert 'city' not in out.columns
    assert 'type_Condo' in out.columns
    assert 'city_Concord' in out.columns
    assert 'city_Walnut Creek' in out.columns


def test_default_columns_is_type():
    tdc = TypeDummyCreator()
    assert tdc.columns == ['type']
